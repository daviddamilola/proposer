"""The Open Targets pack against a fake model and canned URL checks."""
import asyncio
import os
import subprocess

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_engine import FakeModel, call, events_of, proposals, text_reply, texts, tool_reply

from proposer import engine, llm, net, server
from proposer.packs import opentargets
from proposer.packs.opentargets import config as cfg

CONFIG = os.path.join(os.path.dirname(__file__), 'data', 'pis_config.yaml')
TEXT = open(CONFIG, encoding='utf-8').read()
GTEX = 'https://storage.googleapis.com/adult-gtex/annotations/v${gtex_version}/metadata-files/'


@pytest.fixture
def run(monkeypatch):
    """One turn with a fake model. URLs containing 'missing' or 'v12' are 404,
    the private buckets answer 403, anything else exists."""
    monkeypatch.setenv('PROPOSER_URL', 'http://model.test/v1')
    monkeypatch.setenv('PROPOSER_MODEL', 'qwen3:8b')
    monkeypatch.setattr(llm, '_omit_reasoning', False)
    checked = []

    async def fake_check(urls):
        out = []
        for url in urls:
            checked.append(url)
            if 'otar007-cosmic' in url or '/ot-team/' in url:
                out.append({'url': url, 'ok': False, 'status': 403, 'size': None,
                            'error': 'access denied (403)'})
            elif 'missing' in url or 'v12' in url:
                out.append({'url': url, 'ok': False, 'status': 404, 'size': None,
                            'error': 'not found (404)'})
            else:
                out.append({'url': url, 'ok': True, 'status': 200, 'size': 1234, 'error': None})
        return out
    monkeypatch.setattr(net, 'check_urls', fake_check)
    app = FastAPI()
    app.include_router(server.router(opentargets.pack(config=CONFIG)), prefix='/api')
    client = TestClient(app)

    def go(fake, message='Add the GTEx subject phenotypes', context=None):
        monkeypatch.setattr(engine, 'transport', httpx.MockTransport(fake))
        response = client.post('/api/assistant', json={
            'messages': [{'role': 'user', 'content': message}], 'context': context or {}})
        assert response.status_code == 200, response.text
        return events_of(response)
    go.checked = checked
    return go


def tool_results(fake, request):
    return [m['content'] for m in fake.requests[request]['messages'] if m['role'] == 'tool']


def applies(patch, tmp_path):
    """The patch applies with git to a repository holding the config."""
    (tmp_path / 'pis').mkdir()
    (tmp_path / 'pis' / 'config.yaml').write_text(TEXT, encoding='utf-8')
    (tmp_path / 'change.patch').write_text(patch, encoding='utf-8')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'apply', 'change.patch'], cwd=tmp_path, check=True)
    return yaml.safe_load((tmp_path / 'pis' / 'config.yaml').read_text(encoding='utf-8'))


# ----------------------------------------------------------------- turns --

def test_prompt_and_read_tools(run):
    fake = FakeModel(
        tool_reply(call(0, 'search_config', {'text': 'GTEx'}),
                   call(1, 'read_step', {'step': 'baseline_expression_raw'}),
                   call(2, 'list_steps', {}), call(3, 'read_step', {'step': 'baseline'})),
        text_reply('GTEx is copied by baseline_expression_raw.'))
    events = run(fake, message='Where does GTEx come from?')
    system = fake.requests[0]['messages'][0]['content']
    assert 'input assistant for the Open Targets pipeline' in system
    assert '(3 steps, 6 version pins)' in system
    assert 'Steps: baseline_expression_raw, evidence_cosmic, reactome' in system
    assert 'UNTRUSTED' in system and "maintainer's review of the patch" in system
    search, step, steps, wrong = tool_results(fake, 1)
    assert 'pin gtex_version: 11' in search
    assert 'step baseline_expression_raw · copy GTEx TPM matrix' in search
    assert 'Step baseline_expression_raw: 4 tasks.' in step
    assert 'for each of: PXD000561, PXD000865, PXD016999' in step
    assert '- reactome: 2 tasks (copy, copy_many)' in steps
    assert wrong.startswith("Error: no step named 'baseline'. Close names: baseline_expression_raw")
    assert texts(events).startswith('GTEx is copied') and proposals(events) == []


def test_tasks_are_checked_and_patched(run, tmp_path):
    fake = FakeModel(
        tool_reply(call(0, 'propose_input_tasks', {
            'step': 'baseline_expression_raw', 'source': 'GTEx portal, Annotations folder',
            'tasks': [
                {'name': 'GTEx subject phenotypes', 'url': GTEX + 'SubjectPhenotypesDS.txt',
                 'destination': 'input/baseline_expression/bulkRNAseq/GTEx/subject_phenotypes.txt'},
                {'name': 'copy GTEx missing file', 'url': GTEX + 'missing.txt',
                 'destination': 'input/baseline_expression/bulkRNAseq/GTEx/missing.txt'},
                {'name': 'copy GTEx sample attributes again', 'url': GTEX + 'SampleAttributesDS.txt',
                 'destination': 'input/baseline_expression/bulkRNAseq/GTEx/again.txt'}]})),
        text_reply('Proposed one new input.'))
    (card,) = proposals(run(fake))
    assert card['kind'] == 'tasks' and card['applicable'] is True
    assert card['title'] == 'Add 1 input to baseline_expression_raw'
    assert card['note'] == '1 reachable; 1 not found, left out; 1 already in the config'
    task = {'name': 'copy GTEx subject phenotypes',      # the task type is put first
            'source': GTEX + 'SubjectPhenotypesDS.txt',  # the pin stays a placeholder
            'destination': 'input/baseline_expression/bulkRNAseq/GTEx/subject_phenotypes.txt'}
    assert card['payload']['tasks'] == [task]
    # The URL was checked with the pin filled in.
    assert GTEX.replace('${gtex_version}', '11') + 'SubjectPhenotypesDS.txt' in run.checked
    assert [link['ok'] for link in card['links']] == [True, False]

    patched = applies(card['payload']['patch'], tmp_path)
    assert patched['steps']['baseline_expression_raw'][-1] == task
    assert len(patched['steps']['baseline_expression_raw']) == 4
    added = [line[1:] for line in card['payload']['patch'].splitlines()
             if line.startswith('+') and not line.startswith('+++')]
    assert added == ['', '    - name: copy GTEx subject phenotypes',
                     '      source: ' + task['source'], '      destination: ' + task['destination']]
    assert tool_results(fake, 1)[0].startswith('Shown: 1 copy task for step baseline_expression_raw')


def test_guessed_urls_send_the_model_back(run):
    fake = FakeModel(
        tool_reply(call(0, 'propose_input_tasks', {
            'step': 'reactome', 'source': 'User',
            'tasks': [{'name': 'copy reactome relations', 'destination': 'input/reactome/rel.txt',
                       'url': 'https://reactome.org/download/current/missing_relations.txt'}]})),
        text_reply('I could not find that file.'))
    events = run(fake)
    assert proposals(events) == []
    told = tool_results(fake, 1)[0]
    assert told.startswith('Error: none of the 1 URLs exist') and 'Do not guess' in told


@pytest.mark.parametrize('task, error', [
    ({'name': 'copy x', 'url': 'https://e.org/v${gtex}/x.tsv', 'destination': 'input/reactome/x.tsv'},
     '${gtex} is not a version pin in the config. Close names: gtex_version.'),
    ({'name': 'copy x', 'url': 'https://e.org/x.tsv', 'destination': 'input/reactome/ReactomePathways.txt'},
     'is already written by step reactome ("copy reactome pathways")'),
    ({'name': 'copy x', 'url': 'https://e.org/x.tsv', 'destination': 'output/x.tsv'},
     'must be a file path under input/'),
    ({'name': 'copy x', 'url': 'https://e.org/x.tsv', 'destination': 'input/../etc/x.tsv'},
     'must be a file path under input/'),
    ({'name': 'copy x', 'url': 'gs://bucket/folder/*.parquet', 'destination': 'input/reactome/x'},
     'is a pattern; a copy task takes one file'),
    ({'name': 'copy x', 'url': 'https://e.org/sets.gmt.zip', 'destination': 'input/reactome/sets.gmt'},
     'is a .zip file and a copy does not unpack it'),
    ({'name': 'copy_many x', 'url': 'https://e.org/x.tsv', 'destination': 'input/reactome/x.tsv'},
     'only copy tasks (one file each) can be proposed, not copy_many'),
    ({'name': 'copy x', 'url': 'ftp://e.org/x.tsv', 'destination': 'input/reactome/x.tsv'},
     'is not an https:// or gs:// URL'),
    ({'name': 'copy reactome pathways', 'url': 'https://e.org/x.tsv', 'destination': 'input/reactome/x.tsv'},
     'two tasks share the name "copy reactome pathways"'),
])
def test_tasks_that_cannot_be_proposed(run, task, error):
    fake = FakeModel(
        tool_reply(call(0, 'propose_input_tasks',
                        {'step': 'reactome', 'source': 'User', 'tasks': [task]})),
        text_reply('No.'))
    assert proposals(run(fake)) == []
    assert error in tool_results(fake, 1)[0]


def test_new_step_with_a_new_pin(run, tmp_path):
    fake = FakeModel(
        tool_reply(call(0, 'propose_input_tasks', {
            'step': 'Evidence DepMap', 'source': 'DepMap downloads page, 26Q3 release',
            'pins': {'depmap_version': '26Q3', 'unused_version': 7},
            'tasks': [
                {'name': 'copy DepMap gene effect',
                 'url': 'https://depmap.org/files/${depmap_version}/CRISPRGeneEffect.csv',
                 'destination': 'input/evidence_depmap/${depmap_version}/gene_effect.csv'},
                {'name': 'copy DepMap private models', 'url': 'gs://ot-team/depmap/Model.csv',
                 'destination': 'input/evidence_depmap/model.csv'}]})),
        text_reply('Done.'))
    (card,) = proposals(run(fake))
    assert card['title'] == 'Add 2 inputs to new step evidence_depmap'
    assert card['payload']['pins'] == {'depmap_version': '26Q3'}      # the unused pin is left out
    assert card['payload']['unchecked'] == ['gs://ot-team/depmap/Model.csv']
    assert '1 not checked (private bucket' in card['note'] and 'adds pin depmap_version=26Q3' in card['note']
    assert 'https://depmap.org/files/26Q3/CRISPRGeneEffect.csv' in run.checked
    assert 'https://storage.googleapis.com/ot-team/depmap/Model.csv' in run.checked
    assert [link['url'] for link in card['links']] == [
        'https://depmap.org/files/26Q3/CRISPRGeneEffect.csv']
    patched = applies(card['payload']['patch'], tmp_path)
    assert patched['scratchpad']['depmap_version'] == '26Q3'
    assert list(patched['steps'])[-1] == 'evidence_depmap'
    assert [t['name'] for t in patched['steps']['evidence_depmap']] == [
        'copy DepMap gene effect', 'copy DepMap private models']
    patch = card['payload']['patch']
    assert "+  depmap_version: '26Q3'" in patch          # quoted, as the file writes versions
    assert '+  #: EVIDENCE_DEPMAP STEP :####' in patch


def test_pin_bump_checks_the_sources_that_use_it(run, tmp_path):
    fake = FakeModel(
        tool_reply(call(0, 'propose_version_pin',
                        {'key': 'gtex_version', 'value': 10, 'source': 'GTEx portal, release notes'})),
        text_reply('Proposed GTEx v10.'))
    (card,) = proposals(run(fake, message='Move GTEx to v10'))
    assert card['kind'] == 'pin' and card['applicable'] is True
    assert card['note'].startswith('11 → 10; used by 3 tasks; 2 of 2 checked sources exist')
    assert '2 not checked' in card['note']            # two PRIDE files in a private bucket
    assert 'https://storage.googleapis.com/ot-team/baseline_expression/PRIDE/v10/PXD000561_ppb.json' \
        in run.checked
    patched = applies(card['payload']['patch'], tmp_path)
    assert patched['scratchpad']['gtex_version'] == '10'
    changed = [l for l in card['payload']['patch'].splitlines() if l[:1] in '+-' and l[1:2] == ' ']
    assert changed == ["-  gtex_version: '11'", "+  gtex_version: '10'"]


def test_pin_whose_sources_are_not_there(run):
    fake = FakeModel(
        tool_reply(call(0, 'propose_version_pin',
                        {'key': 'GTEx version', 'value': '12', 'source': 'User'})),
        text_reply('v12 is not published.'))
    (card,) = proposals(run(fake))
    assert card['applicable'] is False and '2 not found at the new value' in card['note']
    assert [link['ok'] for link in card['links']] == [False, False]
    assert 'cannot be applied' in tool_results(fake, 1)[0]


def test_pin_the_config_says_not_to_move(run):
    fake = FakeModel(
        tool_reply(call(0, 'propose_version_pin',
                        {'key': 'cosmic_version', 'value': '2026-05-01', 'source': 'COSMIC site'}),
                   call(1, 'propose_version_pin',
                        {'key': 'cosmic', 'value': '2026-05-01', 'source': 'COSMIC site'})),
        text_reply('The config forbids it.'))
    (card,) = proposals(run(fake))
    assert card['applicable'] is False
    assert card['note'].startswith('the config says not to change this pin')
    assert 'our licence only allows 22.02' in card['payload']['comment']
    first, second = tool_results(fake, 1)
    assert 'Do not bump it' in first
    assert second.startswith("Error: no version pin named 'cosmic'. Close names: cosmic_version")


def test_accepted_cards_build_on_each_other():
    """What the terminal client does: apply() keeps a working copy in the
    context, later proposals are made against it, export() gives one patch."""
    pack = opentargets.pack(config=CONFIG)
    context = {'config_text': TEXT, 'config_original': TEXT}
    pin = {'kind': 'pin', 'title': 'Set gtex_version to 10',
           'payload': {'key': 'gtex_version', 'value': '10'}}
    tasks = {'kind': 'tasks', 'title': 'Add 1 input to reactome', 'payload': {
        'step': 'reactome', 'pins': {}, 'tasks': [
            {'name': 'copy reactome relations: all', 'source': 'https://e.org/a b#c.txt',
             'destination': 'input/reactome/relations.txt'}]}}
    assert pack.apply(pin, context) == 'Added to the patch: Set gtex_version to 10.'
    pack.apply(tasks, context)
    with pytest.raises(ValueError):       # the same destination twice
        pack.apply(tasks, context)
    (name, patch), = pack.export(context).items()
    assert name == 'opentargets-inputs.patch' and patch.startswith('--- a/pis/config.yaml')
    after = cfg.parse(context['config_text'])
    assert after.pins['gtex_version'] == '10'
    assert after.steps['reactome'][-1]['name'] == 'copy reactome relations: all'   # quoted
    assert after.steps['reactome'][-1]['source'] == 'https://e.org/a b#c.txt'
    assert after.steps['reactome'][-2]['name'] == 'copy_many reactome snapshots'
    assert context['applied'] == ['Set gtex_version to 10', 'Add 1 input to reactome']
    assert pack.export({'config_text': TEXT, 'config_original': TEXT}) == {}


# --------------------------------------------------------------- editing --

def test_edits_keep_comments_and_layout():
    out = cfg.add_tasks(TEXT, 'evidence_cosmic', [
        {'name': 'copy cosmic hallmarks', 'source': 'gs://otar007-cosmic/h.json',
         'destination': 'input/evidence/cosmic/hallmarks.json'}])
    lines = out.split('\n')
    at = lines.index('    - name: copy cosmic hallmarks')
    assert lines[at - 1] == '' and lines[at - 2].startswith('      destination: input/evidence/cosmic/cosmic')
    assert lines[at + 3].startswith('  ####')         # still inside the step's banner
    assert out.count('\n') == TEXT.count('\n') + 4 and out.endswith('#\n')
    assert '# Do not bump it.' in cfg.set_pins(out, {'efo_version': '3.96.0', 'new_pin': 'abc'})
    assert cfg.comment_above(TEXT, 'cosmic_version').endswith('See opentargets/issues#4536.')
    assert cfg.comment_above(TEXT, 'gtex_version') == ''
    assert cfg.scalar('11') == "'11'" and cfg.scalar('a: b') == "'a: b'" and cfg.scalar('v1.2') == 'v1.2'
    with pytest.raises(cfg.ConfigError):
        cfg.set_pins(TEXT, {'Bad Key': '1'})


def test_check_targets():
    assert opentargets.check_target('gs://b/a/c.txt') == 'https://storage.googleapis.com/b/a/c.txt'
    assert opentargets.check_target('gs://b/a/*.parquet') is None
    assert opentargets.check_target('https://e.org/${x}') is None
    denied = {'ok': False, 'status': 403}
    assert opentargets.outcome('gs://b/c', denied) == 'unchecked'
    assert opentargets.outcome('https://e.org/c', denied) == 'missing'
    assert opentargets.outcome('gs://b/c', {'ok': False, 'status': 404}) == 'missing'
    assert asyncio.run(opentargets.check_sources([])) == {}
