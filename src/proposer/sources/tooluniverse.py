"""ToolUniverse as a source of read tools and skills.

ToolUniverse (github.com/mims-harvard/ToolUniverse) is a catalogue of a few
thousand tools over scientific databases, each defined in JSON by a name, a
description and a JSON schema, with SKILL.md files that describe research
methods in terms of those tools.

    catalogue = Catalogue.installed()            # or Catalogue('/path/to/ToolUniverse')
    tool  = catalogue.tool('Zenodo_search_records')
    skill = catalogue.skill('tooluniverse-dataset-discovery',
                            sections=['search strategy'], limit=8)

A pack pins the result (Pack.skills). Three things keep this within what an
assistant that only reads and proposes may do:

- Only tools the pack names, or that a pinned skill names, are ever offered;
  the model cannot search the catalogue or run a tool by name.
- Tool types that run code, call other models, start remote compute or write
  files are refused (DENIED_TYPES), whoever names them.
- Results are external text: they are cut to RESULT_CHARS and marked as
  untrusted, like the text of a paper.

Reading the catalogue needs only its JSON files. Running a tool needs the
`tooluniverse` package (the `tooluniverse` extra of this project).
"""
import asyncio
import glob
import html
import importlib.util
import json
import os
import re

from pydantic import BaseModel

from .. import read, skill
from ..pack import Tool
from ..proposal import clip

RESULT_CHARS = 6000          # per tool result
STRING_CHARS = 240           # per string in a result
LIST_ITEMS = 15              # per list in a result
DESCRIPTION_CHARS = 400      # per tool, in the schema sent to the model
PARAMETER_CHARS = 160        # per parameter description
RUN_TIMEOUT = 60

# Tool types that are not a read of a public source.
DENIED_TYPES = frozenset({
    'AgenticTool', 'SmolAgentTool', 'ComposeTool',          # call models, chain tools
    'ExecuteTool', 'PythonCodeExecutor', 'PythonScriptRunner', 'ExecutedNotebookTool',
    'MCPAutoLoaderTool', 'MCPClientTool', 'RemoteTool',     # reach other servers
    'BoltzAPITool',                                         # start paid remote compute
    'FileDownloadTool',                                     # write files
})


class AnyArgs(BaseModel):
    """Arguments passed through; ToolUniverse validates them against its schema."""
    model_config = {'extra': 'allow'}


class Catalogue:
    """The tool definitions and skills of a ToolUniverse checkout or install."""

    def __init__(self, root: str, run=None):
        """root: a ToolUniverse checkout, or a folder of *_tools.json files.
        run: (name, arguments) -> result, instead of the tooluniverse package."""
        root = os.path.expanduser(root)
        nested = os.path.join(root, 'src', 'tooluniverse', 'data')
        self.data = nested if os.path.isdir(nested) else root
        self.skills_root = os.path.join(root, 'skills')
        self.definitions = {}
        for path in sorted(glob.glob(os.path.join(self.data, '*.json'))):
            try:
                with open(path, encoding='utf-8') as handle:
                    items = json.load(handle)
            except (OSError, ValueError):
                continue
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and item.get('name'):
                    self.definitions.setdefault(item['name'], item)
        if not self.definitions:
            raise ValueError('no ToolUniverse tool definitions found in %s' % self.data)
        self._run = run
        self._engine = None
        self._loaded = set()

    @classmethod
    def installed(cls, run=None) -> 'Catalogue':
        """The catalogue of the installed tooluniverse package."""
        spec = importlib.util.find_spec('tooluniverse')
        if spec is None or not spec.submodule_search_locations:
            raise ValueError('the tooluniverse package is not installed '
                             '(install proposer with the tooluniverse extra)')
        return cls(os.path.join(list(spec.submodule_search_locations)[0], 'data'), run)

    # ------------------------------------------------------------ tools --

    def allowed(self, name: str) -> bool:
        definition = self.definitions.get(name)
        return bool(definition) and definition.get('type') not in DENIED_TYPES

    def tool(self, name: str) -> Tool:
        """The ToolUniverse tool `name` as a read tool."""
        definition = self.definitions.get(name)
        if definition is None:
            raise ValueError('ToolUniverse has no tool named %s' % name)
        if definition.get('type') in DENIED_TYPES:
            raise ValueError('%s is a %s, which is not a read of a public source'
                             % (name, definition['type']))

        async def run(args, turn):
            return await self.run(name, args.model_dump())

        return Tool(name, clip(definition.get('description') or name, DESCRIPTION_CHARS),
                    AnyArgs, run, parameters=compact(definition.get('parameter')),
                    status='Asking %s…' % name.split('_')[0], timeout=RUN_TIMEOUT + 10)

    def tools(self, names) -> list[Tool]:
        return [self.tool(name) for name in names]

    async def run(self, name: str, arguments: dict) -> str:
        """Run a tool; its result as bounded, marked text for the model."""
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._execute, name, arguments), RUN_TIMEOUT)
        except asyncio.TimeoutError:
            raise read.ToolError('%s did not answer in %d seconds' % (name, RUN_TIMEOUT))
        except read.ToolError:
            raise
        except Exception as error:      # a tool of someone else's making
            raise read.ToolError('%s failed: %s' % (name, clip(str(error), 300)
                                                    or type(error).__name__))
        if isinstance(result, dict) and result.get('status') == 'error':
            raise read.ToolError('%s: %s' % (name, clip(str(result.get('error') or 'failed'), 400)))
        text = result if isinstance(result, str) else json.dumps(
            condense(result), ensure_ascii=False, separators=(',', ':'), default=str)
        if len(text) > RESULT_CHARS:
            text = text[:RESULT_CHARS] + '\n[… cut; ask for fewer results or a narrower query]'
        return read.UNTRUSTED + text

    def _execute(self, name: str, arguments: dict):
        if self._run is not None:
            return self._run(name, arguments)
        if self._engine is None:
            # It reports its progress on stdout unless told otherwise.
            os.environ.setdefault('TOOLUNIVERSE_LOG_LEVEL', 'WARNING')
            try:
                from tooluniverse import ToolUniverse
            except ImportError:
                raise read.ToolError('the tooluniverse package is not installed on this '
                                     'server, so %s cannot be run' % name)
            self._engine = ToolUniverse()
        if name not in self._loaded:
            self._engine.load_tools(include_tools=[name])
            self._loaded.add(name)
        return self._engine.run_one_function({'name': name, 'arguments': arguments})

    # ----------------------------------------------------------- skills --

    def skill(self, name: str, tools=None, sections=None, instructions=None,
              limit: int = 8, title: str = '') -> skill.Skill:
        """A skill from skills/<name>/SKILL.md of the checkout.

        tools         tool names to bring; default: the allowed tools the skill
                      names, in order of first mention, up to `limit`
        sections      headings of the parts of the method to keep (see
                      skill.sections); default: the whole text, cut to budget
        instructions  the method in the pack's own words, instead of the file's
        """
        path = os.path.join(self.skills_root, name, 'SKILL.md')
        if not os.path.isfile(path):
            raise ValueError('no skill named %s in %s' % (name, self.skills_root))
        meta, body = skill.read_markdown(path)
        if tools is None:
            tools = [n for n in skill.names_in(body, self.definitions)
                     if self.allowed(n)][:limit]
        return skill.Skill(
            name=str(meta.get('name') or name),
            title=title or str(meta.get('name') or name).removeprefix('tooluniverse-')
            .replace('-', ' '),
            instructions=instructions if instructions is not None
            else skill.sections(body, sections),
            tools=self.tools(tools))


def condense(value, depth=0):
    """An API response with what a small model cannot use taken out: markup,
    the tails of long strings and lists, empty fields, deep nesting."""
    if isinstance(value, str):
        text = value
        if '<' in text and '>' in text:
            text = html.unescape(re.sub(r'<[^>]+>', ' ', text))
        return clip(text, STRING_CHARS)
    if depth >= 8:
        return '…'
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            item = condense(item, depth + 1)
            if item not in (None, '', [], {}):
                out[key] = item
        return out
    if isinstance(value, (list, tuple)):
        items = [condense(item, depth + 1) for item in value[:LIST_ITEMS]]
        if len(value) > LIST_ITEMS:
            items.append('… %d more' % (len(value) - LIST_ITEMS))
        return items
    return value


def compact(schema) -> dict:
    """A tool's JSON schema as sent to the model: descriptions cut, defaults,
    examples and nested detail dropped."""
    if not isinstance(schema, dict):
        return {'type': 'object', 'properties': {}, 'required': []}
    properties = {}
    for name, prop in (schema.get('properties') or {}).items():
        if not isinstance(prop, dict):
            continue
        kind = prop.get('type')
        if isinstance(kind, list):       # ["string", "null"]
            kind = next((k for k in kind if k != 'null'), 'string')
        out = {'type': kind or 'string'}
        if prop.get('description'):
            out['description'] = clip(str(prop['description']), PARAMETER_CHARS)
        if isinstance(prop.get('enum'), list) and len(prop['enum']) <= 12:
            out['enum'] = prop['enum']
        if out['type'] == 'array':
            items = prop.get('items') if isinstance(prop.get('items'), dict) else {}
            out['items'] = {'type': items.get('type') or 'string'}
        properties[name] = out
    required = [name for name in schema.get('required') or [] if name in properties]
    return {'type': 'object', 'properties': properties, 'required': required}
