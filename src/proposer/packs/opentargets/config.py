"""Reading and editing the input-stage config of the Open Targets pipeline
(pis/config.yaml in github.com/opentargets/pipeline).

The file has two parts that matter here:

    scratchpad:                    version pins, used as ${name} below
      gtex_version: '11'

    steps:
      baseline_expression_raw:     a step: a list of tasks
        - name: copy GTEx sample attributes
          source: https://…/v${gtex_version}/…
          destination: input/baseline_expression/…/sample_attributes.txt

The first word of a task's name is its type (copy, copy_many, explode, …).

It is read with PyYAML but edited as text, so that comments, banners and
quoting stay as the maintainers wrote them. Every edit is checked by parsing
the result and comparing it with what was meant.
"""
import dataclasses
import difflib
import re

import yaml

KEY = re.compile(r'^[a-z][a-z0-9_]*$')
PLACEHOLDER = re.compile(r'\$\{([^}]+)\}')
BANNER_WIDTH = 100


class ConfigError(Exception):
    """The config cannot be read or edited as asked."""


@dataclasses.dataclass
class Task:
    step: str
    name: str
    sources: list[str]              # `source`, or each of `sources`
    destination: str | None
    loop: list[str] | None = None   # values of an enclosing `foreach` list
    raw: dict = dataclasses.field(default_factory=dict)

    @property
    def kind(self) -> str:
        return (self.name.split() or ['?'])[0]


@dataclasses.dataclass
class Config:
    text: str
    pins: dict[str, str]
    steps: dict[str, list]

    def tasks(self, step: str | None = None) -> list[Task]:
        """Every task, with those nested under foreach/glob (`do`) flattened."""
        out = []
        for name, items in self.steps.items():
            if step is None or name == step:
                _walk(name, items or [], None, out)
        return out


def _walk(step, items, loop, out):
    for item in items:
        if not isinstance(item, dict):
            continue
        sources = item.get('source', item.get('sources'))
        sources = [sources] if isinstance(sources, str) else list(sources or [])
        destination = item.get('destination')
        out.append(Task(step, str(item.get('name', '')),
                        [s for s in sources if isinstance(s, str)],
                        destination if isinstance(destination, str) else None, loop, item))
        if isinstance(item.get('do'), list):
            each = item.get('foreach')
            _walk(step, item['do'],
                  [str(v) for v in each] if isinstance(each, list) else loop, out)


def parse(text: str) -> Config:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError('the config is not valid YAML: %s' % error)
    if not isinstance(data, dict) or not isinstance(data.get('steps'), dict):
        raise ConfigError('the config has no `steps:` section')
    pins = {str(k): str(v) for k, v in (data.get('scratchpad') or {}).items()}
    return Config(text, pins, data['steps'])


def resolve(value: str, pins: dict[str, str]) -> tuple[str, list[str]]:
    """`value` with ${pins} filled in, and the placeholders left unknown."""
    unknown = []

    def fill(match):
        key = match.group(1)
        if key in pins:
            return pins[key]
        unknown.append(key)
        return match.group(0)

    return PLACEHOLDER.sub(fill, value), unknown


# -------------------------------------------------------------- editing --

def scalar(value: str) -> str:
    """`value` as a YAML scalar: plain if it reads back as the same string,
    else single-quoted (the file's style for versions such as '11')."""
    try:
        plain = yaml.safe_load('k: %s' % value) == {'k': value}
    except yaml.YAMLError:
        plain = False
    if plain and '\n' not in value:
        return value
    return "'%s'" % value.replace("'", "''")


def pin_scalar(value: str) -> str:
    """A pin's value as the file writes them: short values that start with a
    digit are quoted ('11', '2026Q3'); names and long hashes are plain."""
    if value[:1].isdigit() and len(value) < 20 and "'" not in value:
        return "'%s'" % value
    return scalar(value)


def _section(lines: list[str], name: str) -> tuple[int, int]:
    """Line range [start, end) of the body of top-level key `name`."""
    for i, line in enumerate(lines):
        if re.match(r'^%s:\s*(#.*)?$' % re.escape(name), line):
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if re.match(r'^\S', lines[j]) and not lines[j].startswith('#'):
                    end = j
                    break
            return i + 1, end
    raise ConfigError('the config has no `%s:` section' % name)


def _last_content(lines, start, end) -> int:
    """Index after the last non-blank line in [start, end)."""
    while end > start and not lines[end - 1].strip():
        end -= 1
    return end


def comment_above(text: str, key: str) -> str:
    """The comment lines directly above a pin, joined: maintainers explain
    there why a pin must not move ("our licence only allows 22.02")."""
    lines = text.split('\n')
    start, end = _section(lines, 'scratchpad')
    for i in range(start, end):
        if re.match(r'^\s+%s:' % re.escape(key), lines[i]):
            found = []
            j = i - 1
            while j >= start and lines[j].strip().startswith('#'):
                found.insert(0, lines[j].strip().lstrip('#').strip())
                j -= 1
            return ' '.join(found)
    return ''


def set_pins(text: str, pins: dict[str, str]) -> str:
    """The config with each pin set: an existing line is rewritten in place,
    a new one is added at the end of the scratchpad."""
    lines = text.split('\n')
    for key, value in pins.items():
        if not KEY.match(key):
            raise ConfigError('%r is not a valid pin name (lower case, digits, _)' % key)
        start, end = _section(lines, 'scratchpad')
        for i in range(start, end):
            match = re.match(r'^(\s+)%s:\s*(.*)$' % re.escape(key), lines[i])
            if match:
                trailing = re.search(r'\s+#.*$', match.group(2))
                lines[i] = '%s%s: %s%s' % (match.group(1), key, pin_scalar(value),
                                           trailing.group(0) if trailing else '')
                break
        else:
            at = _last_content(lines, start, end)
            indent = '  '
            for i in range(start, end):
                found = re.match(r'^(\s+)[A-Za-z0-9_]+:', lines[i])
                if found:
                    indent = found.group(1)
                    break
            lines.insert(at, '%s%s: %s' % (indent, key, pin_scalar(value)))
    out = '\n'.join(lines)
    parsed = parse(out)
    for key, value in pins.items():
        if parsed.pins.get(key) != value:
            raise ConfigError('could not set %s safely' % key)
    return out


def _task_lines(task: dict, indent: str) -> list[str]:
    lines = ['%s- name: %s' % (indent, scalar(task['name']))]
    for field in ('source', 'destination'):
        lines.append('%s  %s: %s' % (indent, field, scalar(task[field])))
    return lines


def add_tasks(text: str, step: str, tasks: list[dict]) -> str:
    """The config with copy tasks ({name, source, destination}) added at the
    end of `step`; the step is created at the end of the file if it is new."""
    if not KEY.match(step):
        raise ConfigError('%r is not a valid step name (lower case, digits, _)' % step)
    before = parse(text)
    written = {task.destination for task in before.tasks()}
    names = {task.name for task in before.tasks(step)}
    for task in tasks:
        if task['destination'] in written:
            raise ConfigError('%s is already written by a task' % task['destination'])
        if task['name'] in names:
            raise ConfigError('step %s already has a task named "%s"' % (step, task['name']))
        written.add(task['destination'])
        names.add(task['name'])
    lines = text.split('\n')
    start, end = _section(lines, 'steps')
    header = None
    for i in range(start, end):
        if re.match(r'^  %s:\s*(#.*)?$' % re.escape(step), lines[i]):
            header = i
            break
    banners = any(line.startswith('  #: ') for line in lines[start:end])
    block = []
    for task in tasks:
        block += _task_lines(task, '    ') + ['']
    block.pop()

    if header is not None:
        stop = end
        for j in range(header + 1, end):
            # The step ends at the next step, its banner, or the closing rule.
            if re.match(r'^  [A-Za-z0-9_]+:\s*(#.*)?$', lines[j]) or \
                    re.match(r'^  #(:|#{5,})', lines[j]):
                stop = j
                break
        at = _last_content(lines, header + 1, stop)
        empty = at == header + 1
        lines[at:at] = block if empty else [''] + block
    else:
        at = _last_content(lines, start, end)
        new = ['']
        if banners:
            title = '  #: %s STEP :' % step.upper()
            new.append(title + '#' * max(3, BANNER_WIDTH - len(title)))
        new.append('  %s:' % step)
        new += block
        if banners:
            new.append('  ' + '#' * (BANNER_WIDTH - 2))
        lines[at:at] = new

    out = '\n'.join(lines)
    after = parse(out)
    expected = list(before.steps.get(step) or []) + [
        {'name': t['name'], 'source': t['source'], 'destination': t['destination']}
        for t in tasks]
    others_same = all(after.steps.get(name) == items for name, items in before.steps.items()
                      if name != step)
    if after.steps.get(step) != expected or not others_same or after.pins != before.pins:
        raise ConfigError('could not add the tasks to %s safely' % step)
    return out


def diff(old: str, new: str, path: str) -> str:
    """A unified diff that `git apply` accepts in the pipeline repository."""
    return ''.join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        'a/' + path, 'b/' + path))
