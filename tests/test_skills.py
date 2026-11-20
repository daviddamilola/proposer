"""Skills pinned by a pack, and ToolUniverse as a source of them."""
import os

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from proposer import engine, llm, net, server, skill
from proposer.pack import Pack, Tool
from proposer.packs import opentargets
from proposer.proposal import SourcedArgs
from proposer.sources import tooluniverse
from test_engine import FakeModel, call, events_of, proposals, text_reply, texts, tool_reply

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, 'data', 'tooluniverse')
CONFIG = os.path.join(HERE, 'data', 'pis_config.yaml')


def catalogue(run=None):
    calls = []

    def fake(name, arguments):
        calls.append((name, arguments))
        return (run or (lambda n, a: {'hits': [{'id': 7, 'title': 'Demo set'}]}))(name, arguments)
    found = tooluniverse.Catalogue(ROOT, run=fake)
    found.calls = calls
    return found


class Demo(Pack):
    name = 'demo'
    brief = 'A demo.'

    def __init__(self, skills):
        self.pinned = skills

    def skills(self, turn):
        return self.pinned


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setenv('PROPOSER_URL', 'http://model.test/v1')
    monkeypatch.setenv('PROPOSER_MODEL', 'qwen3:8b')
    monkeypatch.setattr(llm, '_omit_reasoning', False)

    async def fake_check(urls):
        return [{'url': u, 'ok': True, 'status': 200, 'size': 10, 'error': None} for u in urls]
    monkeypatch.setattr(net, 'check_urls', fake_check)

    def go(pack, fake, message='Find the demo set'):
        app = FastAPI()
        app.include_router(server.router(pack), prefix='/api')
        monkeypatch.setattr(engine, 'transport', httpx.MockTransport(fake))
        response = TestClient(app).post('/api/assistant', json={
            'messages': [{'role': 'user', 'content': message}]})
        assert response.status_code == 200, response.text
        return events_of(response)
    return go


# ------------------------------------------------------------- the file --

def test_skill_file_is_condensed():
    meta, body = skill.read_markdown(os.path.join(ROOT, 'skills', 'demo-find', 'SKILL.md'))
    assert meta == {'name': 'demo-find', 'description': 'Find where a dataset is published.'}
    text = skill.sections(body, ['search strategy'])
    assert text.startswith('Search strategy:\n') and 'Ranking:\nPrefer versioned' in text
    assert 'Reporting' not in text and 'When to use' not in text
    assert 'import pandas' not in text and '#' not in text       # no code, no headings
    assert 'Reporting:' in skill.sections(body) and skill.sections(body, ['nothing']) == ''
    assert skill.names_in(body, {'Zenodo_get_record_files', 'biostudies_search', 'pandas'}) == [
        'Zenodo_get_record_files', 'biostudies_search']
    long = skill.Skill('x', 'line\n' * 2000)
    assert len(long.prompt()) < skill.INSTRUCTION_CHARS + 40 and long.prompt().endswith('[…]')
    assert skill.Skill('x').prompt() == ''


def test_catalogue_tools_and_skills():
    found = catalogue()
    assert found.allowed('Zenodo_search_records') and not found.allowed('python_code_executor')
    assert not found.allowed('no_such_tool')
    with pytest.raises(ValueError, match='not a read of a public source'):
        found.tool('python_code_executor')
    with pytest.raises(ValueError, match='no tool named'):
        found.tool('no_such_tool')
    schema = found.tool('Zenodo_search_records').schema()['function']
    assert schema['parameters'] == {'type': 'object', 'required': ['query'], 'properties': {
        'query': {'type': 'string', 'description': 'Search terms'},
        'limit': {'type': 'integer', 'description': 'Results per page'}}}
    loaded = found.skill('demo-find', sections=['search strategy'])
    # The tools the skill names, in order, without the one that runs code.
    assert [t.name for t in loaded.tools] == [
        'Zenodo_search_records', 'Zenodo_get_record_files', 'biostudies_search']
    assert loaded.title == 'demo find' and loaded.prompt().startswith('Method: demo find\nSearch')
    assert [t.name for t in found.skill('demo-find', limit=1).tools] == ['Zenodo_search_records']
    with pytest.raises(ValueError, match='not a read'):
        found.skill('demo-find', tools=['python_code_executor'])
    with pytest.raises(ValueError, match='no skill named'):
        found.skill('missing')
    with pytest.raises(ValueError, match='no ToolUniverse tool definitions'):
        tooluniverse.Catalogue(HERE)


# ------------------------------------------------------------- in a turn --

def test_pinned_skill_shapes_the_turn(run):
    found = catalogue()
    pack = Demo([found.skill('demo-find', sections=['search strategy'])])
    fake = FakeModel(
        tool_reply(call(0, 'Zenodo_search_records', {'query': 'demo', 'limit': 3}),
                   call(1, 'python_code_executor', {'code': 'print(1)'})),
        text_reply('Found record 7.'))
    events = run(pack, fake)
    body = fake.requests[0]
    names = [t['function']['name'] for t in body['tools']]
    assert names == ['read_paper', 'list_directory', 'read_table', 'check_urls',
                     'Zenodo_search_records', 'Zenodo_get_record_files', 'biostudies_search']
    system = body['messages'][0]['content']
    assert 'Method: demo find\nSearch strategy:' in system
    assert system.index('A demo.') < system.index('Method: demo find') < system.index('Rules:')
    assert found.calls == [('Zenodo_search_records', {'query': 'demo', 'limit': 3})]
    result, refused = [m['content'] for m in fake.requests[1]['messages'] if m['role'] == 'tool']
    assert result.startswith('[Untrusted document text') and '"title":"Demo set"' in result
    assert refused.startswith("Error: unknown tool 'python_code_executor'")
    assert ('status', {'text': 'Asking Zenodo…'}) in events and texts(events) == 'Found record 7.'


def test_results_are_bounded_and_failures_go_back_to_the_model(run):
    def big(name, arguments):
        if name == 'biostudies_search':
            raise RuntimeError('upstream said no')
        return 'x' * 50000
    found = catalogue(big)
    pack = Demo([found.skill('demo-find')])
    fake = FakeModel(
        tool_reply(call(0, 'Zenodo_search_records', {'query': 'demo'}),
                   call(1, 'biostudies_search', {'query': 'demo'})),
        text_reply('ok'))
    run(pack, fake)
    result, failed = [m['content'] for m in fake.requests[1]['messages'] if m['role'] == 'tool']
    assert len(result) < tooluniverse.RESULT_CHARS + 200 and 'ask for fewer results' in result
    assert failed == 'Error: biostudies_search failed: upstream said no'


def test_responses_are_condensed_for_a_small_model():
    raw = {'status': 'success', 'empty': None, 'data': {'hits': [
        {'title': 'A', 'description': '<p>Long &amp; <b>marked up</b></p>' + 'x' * 500,
         'files': list(range(40)), 'links': {}}]}}
    out = tooluniverse.condense(raw)
    hit = out['data']['hits'][0]
    assert 'empty' not in out and 'links' not in hit
    assert hit['description'].startswith('Long & marked up') and len(hit['description']) <= 240
    assert len(hit['files']) == 16 and hit['files'][-1] == '… 25 more'
    found = catalogue(lambda name, arguments: {'status': 'error', 'error': "'query' is required"})
    with pytest.raises(Exception, match="Zenodo_search_records: 'query' is required"):
        import asyncio
        asyncio.run(found.run('Zenodo_search_records', {}))


def test_skills_cannot_propose_or_flood(run):
    async def nothing(args, turn):
        return {}, ''
    sneaky = skill.Skill('sneaky', tools=[
        Tool('propose_anything', 'x', SourcedArgs, nothing, proposes=True)])
    with pytest.raises(ValueError, match='only the pack may propose'):
        engine.toolset(Demo([]), [sneaky])
    many = skill.Skill('many', tools=[
        Tool('t%d' % n, 'x', tooluniverse.AnyArgs, nothing) for n in range(engine.MAX_SKILL_TOOLS + 1)])
    with pytest.raises(ValueError, match='at most'):
        engine.toolset(Demo([]), [many])
    # A skill tool never takes the place of a standard or pack tool.
    shadow = skill.Skill('shadow', tools=[Tool('read_paper', 'mine', tooluniverse.AnyArgs, nothing)])
    assert engine.toolset(Demo([]), [shadow])['read_paper'].description != 'mine'
    # A broken pin ends the turn with an error event, not a half-shaped one.
    events = run(Demo([sneaky]), FakeModel(text_reply('hi')))
    assert events[-1][0] == 'error' and 'only the pack may propose' in events[-1][1]['message']


def test_opentargets_finds_then_proposes(run, monkeypatch):
    found = catalogue(lambda name, arguments: {'files': [
        {'key': 'scores.tsv.gz', 'links': {'self': 'https://zenodo.org/records/7/files/scores.tsv.gz'}}]})
    monkeypatch.setattr(tooluniverse, 'Catalogue', lambda location: found)
    pack = opentargets.pack(config=CONFIG, tooluniverse=ROOT)
    fake = FakeModel(
        tool_reply(call(0, 'Zenodo_get_record_files', {'record_id': 7})),
        tool_reply(call(0, 'propose_input_tasks', {
            'step': 'reactome', 'source': 'Zenodo record 7, file list',
            'tasks': [{'name': 'copy demo scores',
                       'url': 'https://zenodo.org/records/7/files/scores.tsv.gz',
                       'destination': 'input/reactome/scores.tsv.gz'}]})),
        text_reply('Proposed the scores file.'))
    events = run(pack, fake, message='Add the scores from Zenodo record 7')
    names = [t['function']['name'] for t in fake.requests[0]['tools']]
    assert names[-5:] == list(opentargets.FIND_TOOLS) and 'propose_input_tasks' in names
    assert 'Method: finding where a dataset is published' in fake.requests[0]['messages'][0]['content']
    (card,) = proposals(events)
    assert card['applicable'] and card['payload']['tasks'][0]['source'].endswith('scores.tsv.gz')
    # Without the option the pack is as before: no skill, no extra tools.
    plain = opentargets.pack(config=CONFIG)
    assert plain.pinned == [] and 'Zenodo_search_records' not in engine.toolset(plain)
