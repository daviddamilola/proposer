"""Tersect pack: describing a genomics dataset for Tersect Browser.

The thing described is the wizard form of the dataset builder: a reference
genome (FASTA, optional GFF3), one VCF per accession, and build options. The
client sends the form as context:

    {"step": "accessions", "form": {...}, "build": {...} | null}

and applies accepted cards to the form itself, so apply() is not implemented.
"""
import re

from ... import read
from ...pack import Pack, Tool, Turn
from ...proposal import clip
from . import propose

S = {'type': 'string'}
SOURCE = {'type': 'string', 'description': 'Where the values came from, e.g. '
          '"Data availability, para. 2 · Table S1" or a folder URL. Required.'}


def _parameters(properties, required):
    return {'type': 'object', 'properties': properties, 'required': required}


# --------------------------------------------------------------- papers --

KEYWORDS = re.compile(
    r'vcf|variant[- ]call|\bsnps?\b|\bindels?\b|genotyp|re-?sequenc|sequenc|'
    r'reference genome|genome assembly|\bassembl|accessions?\b|cultivars?|germplasm|'
    r'\blines?\b|varieties|\bbwa\b|gatk|samtools|bcftools|freebayes|deepvariant|'
    r'deposited|available (at|from|in)|availability|\bftp\b|https?://|\bsra\b|\bena\b|'
    r'bioproject|prj[nde][abc]?\d+|fastq|\breads\b|aligned|mapped', re.IGNORECASE)
METHODS = re.compile(r'method|material|experimental procedure|data description|'
                     r'sequencing|data generation|variant|genotyp', re.IGNORECASE)
TABLE_TOPIC = re.compile(r'accession|sample|\blines?\b|cultivar|variet|genotype|germplasm|'
                         r'strain|individual|isolate|ecotype|landrace|species', re.IGNORECASE)
READ_ARCHIVES = re.compile(r'\b(PRJ(?:NA|EB|DB|CA)\d+|[SED]RP\d{5,}|[SED]RA\d{5,}|'
                           r'[SED]RR\d{5,}|GSE\d{3,}|SAM[NED][A-Z]?\d+)\b')


def signals(text: str) -> list[str]:
    """One-line hints for the model: VCFs, read archives, file types."""
    lines = []
    vcf = len(re.findall(r'\bvcf\b|variant call format', text, re.IGNORECASE))
    fastq = len(re.findall(r'\bfastq\b|raw reads|sequence read archive|\bsra\b', text, re.IGNORECASE))
    archives = []
    for match in READ_ARCHIVES.findall(text):
        if match not in archives:
            archives.append(match)
    if not vcf and (fastq or archives):
        # Small models miss this when it is only implied by the counts below.
        lines.append('VERDICT: this paper appears to publish raw reads only, no VCFs. Say so '
                     'in your first sentence and propose no dataset unless the user gives '
                     'VCF locations.')
    lines.append('VCF mentioned: %s' % ('yes (%d times)' % vcf if vcf else 'no'))
    if fastq:
        lines.append('Raw reads / FASTQ / SRA mentioned: %d times' % fastq)
    if archives:
        lines.append('Read or sample archive ids: %s%s' % (
            ', '.join(archives[:12]), ' …' if len(archives) > 12 else ''))
    return lines


# ---------------------------------------------------------------- state --

def _names(items, limit):
    items = [str(i) for i in items if i]
    if len(items) <= limit:
        return ', '.join(items)
    return '%s … (%d in all)' % (', '.join(items[:limit]), len(items))


def _ref_text(ref):
    if not ref:
        return 'none'
    return '%s (%s)' % (ref.get('name') or ref.get('location') or '?', ref.get('source', '?'))


def summarise_form(form: dict | None) -> list[str]:
    """A compact, bounded description of the wizard form."""
    if not form:
        return ['The form is empty.']
    lines = []
    if form.get('name') or form.get('description'):
        lines.append('Dataset: %s%s' % (form.get('name') or '(no name)',
                                        ' – ' + clip(form['description'], 150)
                                        if form.get('description') else ''))
    reference = form.get('reference') or {}
    sequences = [s.get('name') for s in reference.get('sequences') or []]
    chromosomes = reference.get('chromosomes') or []
    lines.append('Reference: id %s; FASTA %s; GFF3 %s' % (
        reference.get('id') or '(none)', _ref_text(reference.get('fasta')),
        _ref_text(reference.get('gff'))))
    if sequences:
        lines.append('  FASTA sequences (%d): %s' % (len(sequences), _names(sequences, 8)))
    if chromosomes:
        lines.append('  Chromosomes chosen (%d): %s' % (len(chromosomes), _names(chromosomes, 8)))
    targets = chromosomes or sequences
    vcfs = form.get('vcfs') or []
    if vcfs:
        lines.append('VCF files (%d):' % len(vcfs))
        for vcf in vcfs[:12]:
            contigs = vcf.get('contigs') or []
            rename = vcf.get('renameContigs') or {}
            parts = [_ref_text(vcf.get('file'))]
            if vcf.get('samples') is not None:
                parts.append('%d samples' % len(vcf['samples']))
            if vcf.get('inspectError'):
                parts.append('error: ' + clip(vcf['inspectError'], 120))
            if contigs:
                parts.append('contigs %s' % _names(contigs, 4))
                renamed = [rename.get(c, c) for c in contigs]
                if targets and not set(renamed) & set(targets):
                    parts.append('CONTIGS DO NOT MATCH THE REFERENCE')
            if rename:
                parts.append('rename map applied (%d)' % len(rename))
            lines.append('  - ' + '; '.join(parts))
        if len(vcfs) > 12:
            lines.append('  … %d more VCF files' % (len(vcfs) - 12))
    accessions = form.get('accessions') or []
    if accessions:
        groups = sorted({a.get('group') for a in accessions if a.get('group')})
        lines.append('Accessions (%d): %s%s' % (
            len(accessions), _names([a.get('name') for a in accessions], 15),
            '; groups: ' + _names(groups, 6) if groups else ''))
    else:
        lines.append('Accessions: none yet')
    options = form.get('options') or {}
    if options:
        lines.append('Options: effects %s%s; types %s; homozygous %s%s' % (
            options.get('effects'),
            ' (%s)' % options['snpeffDb'] if options.get('snpeffDb') else '',
            options.get('types'), 'yes' if options.get('homozygous') else 'no',
            '; uses an existing index' if options.get('useIndex') else ''))
    return lines


def summarise_build(build: dict | None) -> list[str]:
    if not build:
        return []
    lines = ['Build %s (status %s):' % (build.get('id'), build.get('status'))]
    error = build.get('error') or {}
    if error:
        where = ', '.join('%s %s' % (k, error[k]) for k in ('accession', 'file', 'chromosome')
                          if error.get(k))
        lines.append('  Error at stage %s%s: %s' % (error.get('stage'),
                                                    ' (%s)' % where if where else '',
                                                    clip(error.get('message'), 600)))
        fix = error.get('fix') or {}
        if fix:
            lines.append('  Suggested fix: step %s%s%s' % (
                fix.get('step'), ', kind %s' % fix['kind'] if fix.get('kind') else '',
                ', target %s' % fix['target'] if fix.get('target') else ''))
    tail = (build.get('log') or [])[-25:]
    if tail:
        lines.append('  Log tail (! marks error lines):')
        for entry in tail:
            text = entry.get('t') if isinstance(entry, dict) else str(entry)
            flag = '!' if isinstance(entry, dict) and entry.get('error') else ' '
            lines.append('  %s %s' % (flag, clip(text or '', 200)))
    return lines


# ----------------------------------------------------------------- pack --

def _accessions_status(args):
    if args.pattern:
        from urllib.parse import urlparse
        return 'Listing %s and checking the VCF links…' % (
            urlparse(args.base_url or '').hostname or args.base_url or '')
    return 'Checking %d VCF links…' % len(args.rows)


class TersectPack(Pack):
    name = 'tersect'
    role = 'the dataset assistant of Tersect Browser'
    goal = 'fill in the form that describes a dataset to build'
    authority = "the form's own validation"
    brief = """A Tersect dataset needs:
- a reference genome: FASTA (required) and a GFF3 with gene models (optional; needed to \
build variant effects);
- at least 2 accessions, each with one VCF (per-accession files, or samples of a \
multi-sample VCF) aligned to that same reference;
- names matching [A-Za-z0-9][A-Za-z0-9._-]* (the server fixes others; keep the original as \
label).
Options: effects build|prebuilt (snpEff database)|none; types snps|indels|both; \
homozygous true|false."""
    rules = (
        'For a folder of VCFs, call list_directory first, then propose_accessions with '
        'base_url and a pattern (and name_regex) rather than writing every row out; labels '
        'and groups can come later from a table (propose_sheet).',
        'If the paper publishes only raw reads (FASTQ, SRA/ENA/BioProject) and no VCFs, say '
        'so in your first sentence, propose no dataset, and list what is needed first: '
        'aligning the reads to a reference and calling variants (e.g. nf-core/sarek or BWA + '
        'GATK/bcftools), then one VCF per accession. Keep that answer short.',
    )
    paper = read.PaperProfile(keywords=KEYWORDS, methods=METHODS, table_topic=TABLE_TOPIC,
                              signals=signals)

    def tools(self) -> list[Tool]:
        return [
            Tool('propose_reference', 'Propose the reference genome (FASTA, optional GFF3) and '
                 'the dataset name/description. Links are checked before the user sees them.',
                 propose.ReferenceArgs, propose.propose_reference, proposes=True,
                 kind='reference', status='Checking the reference links…',
                 parameters=_parameters(
                     {'fasta_url': S, 'gff_url': S, 'reference_id': S, 'name': S,
                      'description': S, 'source': SOURCE}, ['source'])),
            Tool('propose_accessions', 'Propose accession rows, one VCF each (or a sample of a '
                 'multi-sample VCF). For a folder of per-accession VCFs give base_url + pattern '
                 '(e.g. "*.vcf.gz"): every matching file becomes a row, named by name_regex\'s '
                 'first group (e.g. "^(RF_\\d+)") or the file name. Otherwise give rows; vcf '
                 'may be a file name relative to base_url. Put accessions without a known VCF '
                 'in missing.',
                 propose.AccessionsArgs, propose.propose_accessions, proposes=True,
                 kind='accessions', status=_accessions_status,
                 parameters=_parameters(
                     {'base_url': S, 'pattern': S, 'name_regex': S,
                      'group': {'type': 'string',
                                'description': 'Group for folder rows, e.g. a species'},
                      'rows': {'type': 'array', 'items': {'type': 'object', 'properties': {
                          'name': S, 'label': S, 'species': S, 'vcf': S, 'sample': S,
                          'info': {'type': 'object'}}, 'required': ['name', 'vcf']}},
                      'missing': {'type': 'array', 'items': S}, 'source': SOURCE},
                     ['source'])),
            Tool('propose_rename_map', 'Propose renaming VCF contigs to the reference sequence '
                 'names. The server works the map out from the form when it can.',
                 propose.RenameMapArgs, propose.propose_rename_map, proposes=True,
                 kind='rename-map', once=True,
                 parameters=_parameters(
                     {'files': {'type': 'array', 'items': S, 'description': 'VCF file names'},
                      'map': {'type': 'object', 'description': 'VCF contig -> reference name'},
                      'source': SOURCE}, ['source'])),
            Tool('propose_options', 'Propose build options.',
                 propose.OptionsArgs, propose.propose_options, proposes=True, kind='options',
                 parameters=_parameters(
                     {'effects': {'type': 'string', 'enum': ['build', 'prebuilt', 'none']},
                      'snpeff_db': S,
                      'types': {'type': 'string', 'enum': ['snps', 'indels', 'both']},
                      'homozygous': {'type': 'boolean'}, 'source': SOURCE}, ['source'])),
            Tool('propose_sheet', 'Propose labels, groups and info for accessions already in '
                 'the form, matched by name (e.g. from a table the user pasted).',
                 propose.SheetArgs, propose.propose_sheet, proposes=True, kind='sheet',
                 parameters=_parameters(
                     {'rows': {'type': 'array', 'items': {'type': 'object', 'properties': {
                         'name': S, 'label': S, 'group': S, 'info': {'type': 'object'}},
                         'required': ['name']}}, 'source': SOURCE}, ['rows', 'source'])),
        ]

    def describe(self, context: dict) -> list[str]:
        lines = ['Current wizard step: %s' % (context.get('step') or 'unknown')]
        lines += summarise_form(context.get('form'))
        build = summarise_build(context.get('build'))
        if build:
            lines += [''] + build + [
                'This conversation is about that build. Explain the error in plain words and '
                'how to fix it. If VCF contig names do not match the reference, call '
                'propose_rename_map (the server works the map out).']
        return lines

    async def start_turn(self, turn: Turn, messages: list) -> None:
        """On a turn about a failed build whose fix is a rename map, show the map
        straight away. It is computed from the form (VCF contigs against the
        reference), so it does not depend on the model calling a tool, which
        qwen3:8b tended to describe instead of call."""
        build = turn.context.get('build') or {}
        fix = (build.get('error') or {}).get('fix') or {}
        if fix.get('kind') != 'rename-map':
            return
        # Skip if an earlier turn already showed a map the user has not dismissed.
        for message in turn.messages:
            if message.role == 'assistant':
                for line in message.content.splitlines():
                    if line.startswith('[Proposal "Rename') and 'dismissed' not in line:
                        return
        # Every VCF whose contigs miss the reference, not only the one that failed
        # first: the others would fail next.
        args = propose.RenameMapArgs(
            source='VCF contig names in the form · reference chromosomes')
        try:
            proposal, summary = await propose.propose_rename_map(args, turn)
        except read.ToolError:
            return     # not enough known (e.g. contigs not read yet); the model explains
        await turn.emit('proposal', proposal)
        turn.shown.add('rename-map')
        messages.append({'role': 'user', 'content': (
            '[Server note] A rename-map card is already shown to the user: %s Explain the '
            'error in plain words and point to that card; do not propose another map.'
            % summary)})


def pack(**options) -> Pack:
    return TersectPack()
