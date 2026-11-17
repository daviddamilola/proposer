"""The pack interface: what a domain supplies to the assistant.

The engine (engine.py) is the same everywhere: a bounded tool-calling loop in
which a model may only read and propose. A pack makes it an assistant for one
thing, by saying:

    what is being described     brief, rules, describe()
    how to look at it           read tools (on top of the standard ones)
    what can be proposed        propose tools, each of which validates the
                                model's values and returns a proposal card
    what the server knows       start_turn(): cards that need no model
    how a card is applied       apply(), export(): only for clients without a
                                form of their own (the terminal client)

A tool is a pydantic model for the arguments and an async function:

    read tool       run(args, turn) -> str                 text for the model
    propose tool    run(args, turn) -> (proposal, summary)  card, and one line
                                                            for the model

Either may raise read.ToolError(message): the message goes back to the model,
which can correct itself and call again.
"""
import dataclasses
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from . import read


@dataclasses.dataclass
class Turn:
    """One user turn, as the tools see it."""
    pack: 'Pack'
    context: dict                      # what the client sent: form, state, …
    emit: Callable                     # async (event, data): to the client
    messages: list = dataclasses.field(default_factory=list)   # the conversation so far
    shown: set = dataclasses.field(default_factory=set)        # kinds of card shown this turn
    state: dict = dataclasses.field(default_factory=dict)      # scratch space for the pack


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable                      # async (args, turn)
    proposes: bool = False             # False: read tool
    kind: str | None = None            # proposal kind, for `once`
    once: bool = False                 # refuse a second card of this kind in a turn
    status: Any = None                 # text, or args -> text, shown while it runs
    parameters: dict | None = None     # JSON schema; worked out from args if None
    timeout: float | None = None       # seconds; the engine's default if None

    def schema(self) -> dict:
        return {'type': 'function', 'function': {
            'name': self.name, 'description': self.description,
            'parameters': self.parameters or compact_schema(self.args)}}

    def status_text(self, args) -> str:
        if callable(self.status):
            return self.status(args)
        return self.status or ('Preparing a suggestion…' if self.proposes else 'Looking…')


def compact_schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema without what a small model does not need:
    titles, defaults, `anyOf: [X, null]` wrappers and $ref indirection."""
    full = model.model_json_schema()
    definitions = full.get('$defs', {})

    def simplify(node):
        if not isinstance(node, dict):
            return node
        if '$ref' in node:
            return simplify(definitions[node['$ref'].rsplit('/', 1)[1]])
        if 'anyOf' in node:
            options = [o for o in node['anyOf'] if o.get('type') != 'null']
            merged = dict(simplify(options[0])) if len(options) == 1 else {}
            if node.get('description'):
                merged['description'] = node['description']
            return merged
        out = {}
        for key, value in node.items():
            if key in ('title', 'default', 'additionalProperties', '$defs'):
                continue
            if key == 'properties':
                out[key] = {name: simplify(prop) for name, prop in value.items()
                            if name not in ('title', 'note')}
            elif key == 'items':
                out[key] = simplify(value)
            else:
                out[key] = value
        return out

    schema = simplify(full)
    schema.setdefault('properties', {})
    schema.setdefault('required', [])
    return schema


class Pack:
    """Subclass and fill in. Only `brief` and tools() are needed for a start."""

    name = 'pack'
    # "You are <role>." — e.g. 'the dataset assistant of Tersect Browser'.
    role = 'an assistant'
    # "You help a user <goal>." — e.g. 'fill in the form that describes a dataset'.
    goal = 'describe what is to be built'
    # What has the last word once the user applies a proposal.
    authority = "the form's own validation"
    # What the thing needs: required parts, name rules, options. A few lines.
    brief = ''
    # Domain rules, one per item. The engine adds its own (rules that hold in
    # every domain: only URLs seen in a tool result, always a source, …).
    rules: tuple[str, ...] = ()
    # How read_paper condenses a paper for this domain.
    paper = read.DEFAULT_PROFILE
    # Standard read tools left out of this pack, by name.
    without: tuple[str, ...] = ()

    def tools(self) -> list[Tool]:
        """The pack's own tools; the standard read tools are added to them."""
        return []

    def describe(self, context: dict) -> list[str]:
        """The current state in a few bounded lines, for the system prompt."""
        return []

    async def start_turn(self, turn: Turn, messages: list) -> None:
        """Before the model runs: load what the tools need into turn.state,
        show cards the server can work out by itself (turn.emit('proposal', …),
        turn.shown.add(kind)), or append a '[Server note] …' user message."""

    def apply(self, proposal: dict, context: dict) -> str:
        """Apply an accepted proposal to `context`; a line saying what changed.
        Called by a client on the user's say-so, never by the model."""
        raise NotImplementedError('Proposals of the %s pack are applied in its own interface.'
                                  % self.name)

    def export(self, context: dict) -> dict[str, str]:
        """Files (name -> text) holding what was applied, for the user to take
        away: a patch, a manifest."""
        return {}
