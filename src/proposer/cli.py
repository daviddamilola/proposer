"""Terminal client.

    proposer chat  --pack opentargets [--set config=pis/config.yaml] [--out DIR] [message]
    proposer serve --pack tersect [--port 8000]

chat runs turns in the terminal. Each proposal is printed as a card; for packs
that can apply cards (Pack.apply) you are asked whether to accept it, and what
you accepted is written to --out when you leave. With a message and no
terminal on stdin it runs that one turn and accepts nothing.

A pack is a built-in name (tersect, opentargets) or module:function, where the
function returns a Pack; --set key=value passes options to it.
"""
import argparse
import asyncio
import importlib
import os
import sys

from . import engine, llm
from .pack import Pack

BUILT_IN = ('tersect', 'opentargets')


def load_pack(name: str, options: dict) -> Pack:
    module, _, attribute = name.partition(':')
    if not attribute:
        module, attribute = 'proposer.packs.' + module, 'pack'
    try:
        factory = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as error:
        raise SystemExit('Cannot load pack %s: %s (built in: %s)'
                         % (name, error, ', '.join(BUILT_IN)))
    return factory(**options)


def format_size(size) -> str:
    if size is None:
        return ''
    for unit in ('B', 'K', 'M', 'G', 'T'):
        if size < 1024 or unit == 'T':
            return ' %s%s' % (('%.1f' % size).rstrip('0').rstrip('.'), unit)
        size /= 1024


def card(proposal: dict, number: int) -> str:
    lines = ['', '┌─ [%d] %s%s' % (number, proposal['title'],
                                   '' if proposal['applicable'] else '   (cannot be applied)'),
             '│ ' + proposal['note'], '│ Source: ' + proposal['source']]
    for link in proposal['links']:
        lines.append('│ %s %s%s' % ('✓' if link['ok'] else '✗', link['url'],
                                    format_size(link.get('size')) if link['ok']
                                    else '  ' + (link.get('error') or 'not reachable')))
    if proposal['more']:
        lines.append('│ ' + proposal['more'])
    patch = (proposal.get('payload') or {}).get('patch')
    if patch:
        lines.append('│')
        lines += ['│ ' + line for line in patch.splitlines()
                  if line[:1] in '+-' and not line.startswith(('+++', '---'))][:40]
    lines.append('└─')
    return '\n'.join(lines)


async def turn(pack: Pack, messages: list, context: dict, think: bool) -> tuple[str, list]:
    """Run one turn, printing as it goes; the answer text and the proposals."""
    said, proposals = [], []
    line_open = {'text': False}

    async def emit(event, data):
        if event == 'text':
            said.append(data['delta'])
            print(data['delta'], end='', flush=True)
            line_open['text'] = True
            return
        if line_open['text']:
            print()
            line_open['text'] = False
        if event == 'status':
            print('· ' + data['text'], file=sys.stderr, flush=True)
        elif event == 'thinking':
            print(data['delta'], end='', file=sys.stderr, flush=True)
        elif event == 'proposal':
            proposals.append(data)
            print(card(data, len(proposals)), flush=True)

    request = engine.Request(messages=messages, context=context, think=think)
    request.context = context      # the pack's working state lives across turns
    async with asyncio.timeout(engine.turn_timeout()):
        await engine.run_turn(pack, request, emit)
    if line_open['text']:
        print()
    return ''.join(said), proposals


def review(pack: Pack, proposals: list, context: dict, interactive: bool) -> list[str]:
    """Ask about each card; the lines recording what the user did with them."""
    record = []
    for proposal in proposals:
        state = 'shown, awaiting the user\'s Accept'
        if interactive and proposal['applicable']:
            answer = input('%s — "%s"? [y/N] ' % (proposal['action'], proposal['title']))
            if answer.strip().lower() in ('y', 'yes'):
                try:
                    print(pack.apply(proposal, context))
                    state = 'accepted by the user'
                except NotImplementedError as error:
                    print(error)
                except ValueError as error:
                    print('Could not apply it: %s' % error)
            else:
                state = 'dismissed by the user'
        record.append('[Proposal "%s": %s]' % (proposal['title'], state))
    return record


def write_out(pack: Pack, context: dict, out: str):
    files = pack.export(context)
    if not files:
        return
    os.makedirs(out, exist_ok=True)
    for name, text in files.items():
        path = os.path.join(out, os.path.basename(name))
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(text)
        print('Wrote %s' % path)


async def chat(args):
    if not llm.configured():
        raise SystemExit('Set PROPOSER_URL to an OpenAI-compatible endpoint, e.g. '
                         'http://localhost:11434/v1 for a local Ollama (and PROPOSER_MODEL).')
    pack = load_pack(args.pack, dict(item.split('=', 1) for item in args.set))
    interactive = sys.stdin.isatty()
    messages, context = [], {}
    first = ' '.join(args.message).strip()
    if not first and not interactive:
        first = sys.stdin.read().strip()
    try:
        while True:
            text = first or (input('\n> ').strip() if interactive else '')
            first = ''
            if not text or text in ('/quit', '/exit'):
                break
            messages.append({'role': 'user', 'content': text})
            try:
                answer, proposals = await turn(pack, messages, context, args.think)
            except (llm.LLMError, engine.BadRequest) as error:
                print('Error: %s' % error, file=sys.stderr)
                messages.pop()
                if not interactive:
                    raise SystemExit(1)
                continue
            record = review(pack, proposals, context, interactive)
            messages.append({'role': 'assistant',
                             'content': '\n'.join([answer or '(no text)'] + record)})
            if not interactive:
                break
    except (EOFError, KeyboardInterrupt):
        print()
    write_out(pack, context, args.out)


def serve(args):
    import uvicorn

    from .server import create_app
    pack = load_pack(args.pack, dict(item.split('=', 1) for item in args.set))
    uvicorn.run(create_app(pack), host=args.host, port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='proposer', description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('chat', 'serve'):
        command = commands.add_parser(name)
        command.add_argument('--pack', required=True,
                             help='built-in name (%s) or module:function' % ', '.join(BUILT_IN))
        command.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                             help='option for the pack; may be repeated')
        if name == 'chat':
            command.add_argument('--out', default='out',
                                 help='folder for what you accepted (default: out)')
            command.add_argument('--think', action='store_true',
                                 help="show the model's reasoning")
            command.add_argument('message', nargs='*')
        else:
            command.add_argument('--host', default='127.0.0.1')
            command.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == 'chat':
        asyncio.run(chat(args))
    else:
        serve(args)


if __name__ == '__main__':
    main()
