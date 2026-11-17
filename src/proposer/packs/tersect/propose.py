"""Proposal tools of the Tersect pack.

A proposal maps onto the wizard form of Tersect Browser (Proposal in
web/src/features/datasets/assistant/types.ts there). The model only supplies
values; the server validates them, checks every URL, leaves unreachable items
out of the payload, decides whether the card can be applied, and words the
counts in the note.
"""
import fnmatch
import posixpath
import re
from typing import Literal
from urllib.parse import unquote, urljoin, urlparse

from pydantic import BaseModel, Field

from ... import net
from ...proposal import SourcedArgs as _Args
from ...proposal import checks as _checks
from ...proposal import clip as _clip
from ...proposal import make as _proposal
from ...proposal import plural as _plural
from ...proposal import sample_links as _sample_links
from ...read import ToolError, list_entries

NAME_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$')
MAX_ROWS = 1000


def fix_name(name: str) -> str:
    """web/src/features/datasets/manifest.ts fixName."""
    fixed = re.sub(r'[^A-Za-z0-9._-]', '_', name.strip())
    if not re.match(r'^[A-Za-z0-9]', fixed):
        fixed = 'S' + fixed
    return fixed[:100]


def chromosome_number(name: str) -> int | None:
    match = re.search(r'(\d+)$', name)
    return int(match.group(1)) if match else None


def suggest_rename_map(contigs: list[str], reference: list[str]) -> dict[str, str]:
    """manifest.ts suggestRenameMap: VCF contig -> reference sequence by trailing number."""
    by_number = {}
    for name in reference:
        number = chromosome_number(name)
        if number is not None and number not in by_number:
            by_number[number] = name
    mapping = {}
    for contig in contigs:
        if contig in reference:
            continue
        number = chromosome_number(contig)
        if number is not None and number in by_number:
            mapping[contig] = by_number[number]
    return mapping


def _file_name(url: str) -> str:
    return unquote(posixpath.basename(urlparse(url).path)) or url


def _url_ref(url: str, check: dict) -> dict:
    return {'source': 'url', 'location': url, 'name': _file_name(url),
            'size': check.get('size')}


# --------------------------------------------------------- arguments --

class ReferenceArgs(_Args):
    fasta_url: str | None = None
    gff_url: str | None = None
    reference_id: str | None = None
    name: str | None = None
    description: str | None = None


class AccessionRowArgs(BaseModel):
    model_config = {'extra': 'ignore'}
    name: str = Field(min_length=1)
    label: str | None = None
    group: str | None = None
    species: str | None = None
    vcf: str = Field(min_length=1, description='VCF URL, or a file name relative to base_url')
    sample: str | None = None
    info: dict[str, str | int | float | bool | None] | None = None


class AccessionsArgs(_Args):
    base_url: str | None = None
    rows: list[AccessionRowArgs] = Field(default_factory=list, max_length=MAX_ROWS)
    missing: list[str] = Field(default_factory=list)
    # Folder mode: every file in base_url matching `pattern` becomes a row, so
    # a small model need not write out (or guess) dozens of file names.
    pattern: str | None = Field(None, max_length=200)
    name_regex: str | None = Field(None, max_length=200)
    group: str | None = Field(None, max_length=100)


# Endings stripped from a VCF file name to get an accession name.
VCF_ENDING = re.compile(r'(\.(snpeff|ann|filtered|pass))*\.(vcf|bcf)(\.gz|\.bgz)?.*$', re.I)


def _name_from_file(file_name: str, name_regex: re.Pattern | None) -> str:
    """RF_001_SZAXPI008746-45.vcf.gz.snpeff.vcf.gz → the regex's first group
    (or match), else the name without its VCF endings."""
    if name_regex:
        match = name_regex.search(file_name)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return VCF_ENDING.sub('', file_name) or file_name


async def _rows_from_folder(args: AccessionsArgs) -> list[AccessionRowArgs]:
    """Rows for the files in args.base_url matching args.pattern."""
    try:
        regex = re.compile(args.name_regex) if args.name_regex else None
    except re.error as error:
        raise ToolError('name_regex is not a valid regular expression: %s' % error)
    url, entries = await list_entries(args.base_url)
    names = [e['name'] for e in entries
             if not e['dir'] and fnmatch.fnmatch(e['name'], args.pattern)]
    if not names:
        raise ToolError('no file in %s matches %r; call list_directory to see the names'
                        % (url, args.pattern))
    explicit = {posixpath.basename(r.vcf) for r in args.rows}
    return [AccessionRowArgs(name=_name_from_file(n, regex), group=args.group,
                             vcf=urljoin(url, n))
            for n in names[:MAX_ROWS] if n not in explicit]


class RenameMapArgs(_Args):
    files: list[str] | None = None
    map: dict[str, str] | None = None
    contigs: list[str] | None = None
    reference: list[str] | None = None


class OptionsArgs(_Args):
    effects: Literal['build', 'prebuilt', 'none'] | None = None
    snpeff_db: str | None = None
    types: Literal['snps', 'indels', 'both'] | None = None
    homozygous: bool | None = None


class SheetRowArgs(BaseModel):
    model_config = {'extra': 'ignore'}
    name: str = Field(min_length=1)
    label: str | None = None
    group: str | None = None
    info: dict[str, str | int | float | bool | None] | None = None


class SheetArgs(_Args):
    rows: list[SheetRowArgs] = Field(min_length=1, max_length=MAX_ROWS)


def _info(values) -> dict[str, str]:
    out = {}
    for key, value in list((values or {}).items())[:30]:
        if value is None or str(key).strip() == '':
            continue
        out[_clip(str(key), 60)] = _clip(str(value), 200)
    return out


def _absolute(url: str, base: str | None) -> str:
    url = url.strip()
    if re.match(r'(https?|ftp)://', url, re.IGNORECASE):
        return url
    if base:
        return urljoin(base if base.endswith('/') else base + '/', url)
    raise ToolError('%s is not a full URL; give a full URL or base_url' % url)


# ----------------------------------------------------------- tools --

async def propose_reference(args: ReferenceArgs, turn):
    urls = [u.strip() for u in (args.fasta_url, args.gff_url) if u and u.strip()]
    checks = await _checks(urls)
    payload = {'fasta': None, 'gff': None,
               'reference_id': None, 'name': None, 'description': None}
    links, problems = [], []
    for key, url in (('fasta', args.fasta_url), ('gff', args.gff_url)):
        if url and url.strip():
            check = checks[url.strip()]
            links.append(check)
            if check['ok']:
                payload[key] = _url_ref(url.strip(), check)
            else:
                problems.append('%s %s' % ('FASTA' if key == 'fasta' else 'GFF',
                                           check.get('error') or 'not reachable'))
    if args.reference_id:
        ref_id = args.reference_id.strip()
        payload['reference_id'] = ref_id if NAME_PATTERN.match(ref_id) else fix_name(ref_id)
    if args.name and args.name.strip():
        payload['name'] = _clip(args.name, 100)
    if args.description and args.description.strip():
        payload['description'] = _clip(args.description, 500)

    usable = payload['fasta'] or payload['gff']
    applicable = bool(usable) if links else any(
        payload[k] for k in ('reference_id', 'name', 'description'))
    parts = []
    if payload['fasta']:
        parts.append('FASTA reachable')
    if payload['gff']:
        parts.append('GFF3 reachable')
    if not args.gff_url:
        parts.append('no gene models given (variant effects need a GFF3)')
    note = '; '.join(parts + problems) or (args.note or 'Dataset details')
    title = args.title or ('Set reference to %s' % payload['reference_id']
                           if payload['reference_id'] else 'Set the reference')
    proposal = _proposal('reference', title, note, args.source, links, None, 'Apply',
                         applicable, payload)
    summary = 'Shown: reference proposal (%s)%s.' % (
        note, '' if applicable else '; it cannot be applied because no link was reachable')
    return proposal, summary


async def propose_accessions(args: AccessionsArgs, turn):
    form = turn.context.get('form')
    if not args.rows and not args.missing and not args.pattern:
        raise ToolError('give rows, or base_url with a pattern (and/or missing names)')
    if args.base_url:
        try:
            net.validate_url(args.base_url)
        except ValueError as error:
            raise ToolError('base_url: %s' % error)
    rows_in = list(args.rows)
    if args.pattern:
        if not args.base_url:
            raise ToolError('pattern needs base_url (the folder to list)')
        rows_in += await _rows_from_folder(args)
    prepared = []
    for row in rows_in:
        prepared.append((row, _absolute(row.vcf, args.base_url)))
    checks = await _checks([url for _, url in prepared])
    if prepared and not any(check['ok'] for check in checks.values()):
        # Small models guess file names ("accession_1.vcf.gz") instead of
        # listing the folder. A card where every link fails helps nobody, so
        # send the model back to look rather than showing one.
        folders = sorted({url.rsplit('/', 1)[0] + '/' for _, url in prepared})
        raise ToolError(
            'none of the %d VCF URLs exist (e.g. %s). Do not guess file names: call '
            'list_directory on %s and propose the exact file names it lists.'
            % (len(prepared), prepared[0][1], ' and '.join(folders[:2])))

    rows, missing, taken, renamed, already = [], [], set(), 0, 0
    existing = {(a.get('name') or '') for a in ((form or {}).get('accessions') or [])}
    for row, url in prepared:
        original = row.name.strip()
        name = original if NAME_PATTERN.match(original) else fix_name(original)
        if name != original:
            renamed += 1
        if name in existing:
            already += 1
            continue
        base, suffix = name, 2
        while name in taken:
            name = '%s_%d' % (base[:95], suffix)
            suffix += 1
        check = checks[url]
        if not check['ok']:
            missing.append(original)
            continue
        taken.add(name)
        label = (row.label or '').strip() or (original if name != original else name)
        rows.append({'name': name, 'label': _clip(label, 100),
                     'group': _clip(row.group or row.species or '', 100) or None,
                     'vcf': _url_ref(url, check),
                     'sample': (row.sample or '').strip() or None,
                     'info': _info(row.info)})
    unreachable = len(missing)
    for name in args.missing:
        name = (name or '').strip()
        if name and name not in missing:
            missing.append(_clip(name, 100))

    ordered = [checks[url] for url in dict.fromkeys(url for _, url in prepared)]
    links, more = _sample_links(ordered)
    no_location = len(missing) - unreachable
    total = len(rows) + len(missing)
    parts = ['%d with reachable VCF links' % len(rows)]
    if unreachable:
        parts.append('%d not reachable' % unreachable)
    if no_location:
        parts.append('%d need a location' % no_location)
    if renamed:
        parts.append('%s renamed to fit the name rule' % _plural(renamed, 'name'))
    if already:
        parts.append('%d already in the form' % already)
    note = '; '.join(parts)
    title = args.title or 'Add %s' % _plural(len(rows) or total, 'accession')
    proposal = _proposal('accessions', title, note, args.source, links, more,
                         'Review %d' % len(rows), bool(rows),
                         {'rows': rows, 'missing': missing})
    failed_names = [m for m in missing[:unreachable]][:5]
    summary = 'Shown: %d of %d accessions with reachable VCFs%s%s.' % (
        len(rows), total,
        ', %d missing' % len(missing) if missing else '',
        ' (unreachable: %s%s)' % (', '.join(failed_names), ' …' if unreachable > 5 else '')
        if unreachable else '')
    if not rows:
        summary += ' Nothing can be applied; check the URLs (list_directory) and try again.'
    return proposal, summary


def _reference_names(form: dict | None) -> list[str]:
    reference = (form or {}).get('reference') or {}
    chosen = reference.get('chromosomes') or []
    return chosen or [s.get('name') for s in reference.get('sequences') or [] if s.get('name')]


async def propose_rename_map(args: RenameMapArgs, turn):
    form = turn.context.get('form')
    vcfs = (form or {}).get('vcfs') or []
    reference = args.reference or _reference_names(form)
    wanted = {f.strip() for f in args.files or [] if f and f.strip()}
    mapping, targets, how = {}, [], None
    if reference:
        sources = []
        for vcf in vcfs:
            name = (vcf.get('file') or {}).get('name') or ''
            if wanted and name not in wanted:
                continue
            if vcf.get('contigs'):
                sources.append((name, vcf['contigs']))
        if args.contigs:
            sources.append((None, args.contigs))
        for name, contigs in sources:
            computed = suggest_rename_map(contigs, reference)
            if computed:
                for contig, target in computed.items():
                    mapping.setdefault(contig, target)
                if name:
                    targets.append(name)
        if mapping:
            how = 'matched by chromosome number against the reference'
    if not mapping and args.map:
        # The model's own map, kept only where it fits what is known.
        known_contigs = {c for v in vcfs for c in v.get('contigs') or []} | set(args.contigs or [])
        for contig, target in args.map.items():
            contig, target = str(contig).strip(), str(target).strip()
            if not contig or not target:
                continue
            if known_contigs and contig not in known_contigs:
                continue
            if reference and target not in reference:
                continue
            mapping[contig] = target
        targets = sorted(wanted) or [(v.get('file') or {}).get('name') for v in vcfs
                                     if set(v.get('contigs') or []) & set(mapping)]
        how = 'as written by the assistant' + (
            '' if known_contigs and reference else '; not checked against the files')
    if not targets and wanted:
        targets = sorted(wanted)
    targets = [t for t in targets if t]
    if not mapping:
        raise ToolError('no rename map could be made: the VCF contigs and reference '
                        'sequences do not share chromosome numbers (or are not known yet). '
                        'Explain the mismatch to the user instead.')
    example = ', '.join('%s → %s' % item for item in list(mapping.items())[:3])
    note = '%s (%s%s); %s' % (_plural(len(mapping), 'contig'), example,
                              ' …' if len(mapping) > 3 else '', how)
    title = args.title or 'Rename %s in %s' % (
        _plural(len(mapping), 'contig'),
        targets[0] if len(targets) == 1 else _plural(len(targets), 'VCF') if targets
        else 'the VCFs')
    proposal = _proposal('rename-map', title, note, args.source, [], None, 'Apply', True,
                         {'files': targets, 'map': mapping})
    return proposal, 'Shown: rename map of %s for %s (%s).' % (
        _plural(len(mapping), 'contig'), ', '.join(targets[:5]) or 'the VCFs', how)


async def propose_options(args: OptionsArgs, turn):
    payload = {'effects': args.effects, 'snpeff_db': (args.snpeff_db or '').strip() or None,
               'types': args.types, 'homozygous': args.homozygous}
    if payload['snpeff_db'] and not payload['effects']:
        payload['effects'] = 'prebuilt'
    if payload['effects'] == 'prebuilt' and not payload['snpeff_db']:
        raise ToolError('effects "prebuilt" needs snpeff_db (the snpEff database name)')
    if all(value is None for value in payload.values()):
        raise ToolError('give at least one of effects, snpeff_db, types, homozygous')
    parts = []
    if payload['effects']:
        parts.append({'build': 'build variant effects from the GFF3',
                      'prebuilt': 'snpEff database %s' % payload['snpeff_db'],
                      'none': 'no variant effects'}[payload['effects']])
    if payload['types']:
        parts.append({'snps': 'SNPs only', 'indels': 'indels only',
                      'both': 'SNPs and indels'}[payload['types']])
    if payload['homozygous'] is not None:
        parts.append('homozygous calls only' if payload['homozygous']
                     else 'all genotype calls')
    note = args.note or '; '.join(parts)
    proposal = _proposal('options', args.title or 'Set build options', note, args.source,
                         [], None, 'Apply', True, payload)
    return proposal, 'Shown: options (%s).' % '; '.join(parts)


async def propose_sheet(args: SheetArgs, turn):
    form = turn.context.get('form')
    accessions = (form or {}).get('accessions') or []
    by_name = {}
    for row in accessions:
        name = row.get('name') or ''
        by_name.setdefault(name.lower(), name)
        by_name.setdefault(fix_name(name).lower(), name)
    rows, unmatched = [], []
    for row in args.rows:
        given = row.name.strip()
        match = by_name.get(given.lower()) or by_name.get(fix_name(given).lower())
        if accessions and not match:
            unmatched.append(given)
            continue
        rows.append({'name': match or given,
                     'label': _clip(row.label, 100) if row.label else None,
                     'group': _clip(row.group, 100) if row.group else None,
                     'info': _info(row.info)})
    if not accessions:
        note = ('No accessions in the form yet; rows apply once accessions with these '
                'names are added')
        applicable = False
    else:
        note = '%d of %d rows match accessions in the form' % (len(rows), len(args.rows))
        if unmatched:
            note += '; not in the form: %s%s' % (', '.join(unmatched[:4]),
                                                 ' …' if len(unmatched) > 4 else '')
        applicable = bool(rows)
    proposal = _proposal('sheet', args.title or 'Fill in %s' % _plural(len(rows), 'accession'),
                         note, args.source, [], None, 'Apply to %d' % len(rows), applicable,
                         {'rows': rows})
    return proposal, 'Shown: sheet with %d rows (%s).' % (len(rows), note)
