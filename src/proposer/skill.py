"""Skills: a method and the read tools that go with it, pinned by a pack.

A skill shapes a turn in two ways: its instructions are added to the system
prompt, and its tools join the tools the model may call. The pack decides
which skills are in force (Pack.skills); the model never chooses one.

Skills only add ways to read. A skill whose tools include a propose tool is
refused: what the assistant can change is decided by the pack alone.

A skill can be written in Python (Skill(...)) or loaded from a SKILL.md file
(a YAML front matter with `name` and `description`, then markdown), the
format used by Claude Code and by ToolUniverse. Those files are written for
large models and run to 10-20 KB, so a pack says which sections it wants and
the result is cut to a budget.
"""
import dataclasses
import re

import yaml

from .pack import Tool

INSTRUCTION_CHARS = 2500     # per skill, in the system prompt


@dataclasses.dataclass
class Skill:
    name: str
    instructions: str = ''                 # the method, in a few paragraphs
    tools: list[Tool] = dataclasses.field(default_factory=list)
    title: str = ''

    def prompt(self) -> str:
        """The skill as a block of the system prompt, or '' if it has no text."""
        text = self.instructions.strip()
        if not text:
            return ''
        if len(text) > INSTRUCTION_CHARS:
            text = text[:INSTRUCTION_CHARS].rsplit('\n', 1)[0].rstrip() + '\n[…]'
        return 'Method: %s\n%s' % (self.title or self.name, text)


def read_markdown(path: str) -> tuple[dict, str]:
    """A SKILL.md file as (front matter, body)."""
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    meta = {}
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.S)
    if match:
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        text = text[match.end():]
    return (meta if isinstance(meta, dict) else {}), text


def sections(body: str, wanted: list[str] | None = None) -> str:
    """The markdown sections whose heading contains one of `wanted`
    (case-insensitive), in document order; the whole body if None. Code blocks
    are dropped: they are for a model that can run code, which this one cannot."""
    body = re.sub(r'```.*?```', '', body, flags=re.S)
    if wanted is None:
        return _tidy(body)
    wanted = [w.lower() for w in wanted]
    kept, keeping, level = [], False, 0
    for line in body.split('\n'):
        heading = re.match(r'^(#{1,6})\s+(.*)$', line)
        if heading:
            depth = len(heading.group(1))
            if any(w in heading.group(2).lower() for w in wanted):
                keeping, level = True, depth
            elif keeping and depth <= level:
                keeping = False
        if keeping:
            kept.append(line)
    return _tidy('\n'.join(kept))


def _tidy(text: str) -> str:
    text = re.sub(r'^#{1,6}\s+(.*)$', r'\1:', text, flags=re.M)     # no headings in prompts
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def names_in(body: str, known) -> list[str]:
    """Tool names from `known` that the text mentions, in order of first mention."""
    known = set(known)
    found = re.findall(r'[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+', body)
    return [name for name in dict.fromkeys(found) if name in known]
