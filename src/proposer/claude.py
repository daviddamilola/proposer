"""Client for the Claude API, as a backend beside the OpenAI-compatible one.

For a server with no GPU to run a local model on. Unlike llm.py this sends the
conversation, and whatever the read tools returned (paper text, file names,
table rows), to Anthropic: an application should offer it only to people who
may decide that, and say so where it is used.

    PROPOSER_CLAUDE_API_KEY        the API key (default: ANTHROPIC_API_KEY)
    PROPOSER_CLAUDE_MODEL          default claude-haiku-5-5
    PROPOSER_CLAUDE_EFFORT         low | medium | high | xhigh | max
                                   (default: the model's own, medium on Haiku)
    PROPOSER_CLAUDE_MAX_TOKENS     per model response (default 16000)
    PROPOSER_CLAUDE_CONTEXT_CHARS  prompt budget in characters (default 300000)

Needs the `anthropic` package (pip install "proposer-llm[claude]").

The module has the interface the engine uses of llm.py: client(),
stream_chat(), model(), configured(), supports_thinking(), prompt_suffix()
and context_chars(). The engine keeps its messages in the chat-completions
shape; they are converted here. The content blocks of an assistant turn that
called tools are kept as they came (`native`) and sent back unchanged, since
the model's thinking blocks are only valid in the turn they were written in.
"""
import json
import os

from . import settings
from .llm import LLMError

try:
    import anthropic
except ImportError:          # the optional dependency is not installed
    anthropic = None

DEFAULT_MODEL = 'claude-haiku-5-5'
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
NOT_RUN = 'Error: not run, too many tool calls in one round. Call it again if needed.'
REFUSED = 'Claude declined to answer this request.'
CUT_OFF = '\n\n[The answer was cut off at the length limit.]'


def installed() -> bool:
    return anthropic is not None


def api_key() -> str:
    return settings.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY', '').strip()


def configured() -> bool:
    return installed() and bool(api_key())


def model() -> str:
    return settings.get('CLAUDE_MODEL', DEFAULT_MODEL)


def supports_thinking() -> bool:
    return True


def prompt_suffix(think: bool) -> str:
    return ''


def _number(name: str, default: int) -> int:
    try:
        return max(1, int(settings.get(name, str(default))))
    except ValueError:
        return default


def context_chars() -> int:
    """Far above what a local model gets: the prompt should not have to be
    shortened within a turn, which would change what thinking blocks follow."""
    return _number('CLAUDE_CONTEXT_CHARS', 300000)


def client(transport=None):
    """An async client; `async with` closes it. `transport` is the engine's
    test hook for llm.py and is not used here (tests replace this function)."""
    return anthropic.AsyncAnthropic(api_key=api_key())


# ----------------------------------------------------------- conversion --

def _tool(schema: dict) -> dict:
    function = schema.get('function') or {}
    return {'name': function.get('name', ''),
            'description': function.get('description', ''),
            'input_schema': function.get('parameters') or {'type': 'object', 'properties': {}}}


def _input(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or '{}')
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _block_type(block) -> str:
    return block.get('type') if isinstance(block, dict) else getattr(block, 'type', '')


def _block_id(block):
    return block.get('id') if isinstance(block, dict) else getattr(block, 'id', None)


def _assistant_blocks(message: dict, thinking: bool) -> list:
    native = message.get('native')
    if native:
        return [block for block in native
                if thinking or _block_type(block) not in ('thinking', 'redacted_thinking')]
    blocks = []
    if (message.get('content') or '').strip():
        blocks.append({'type': 'text', 'text': message['content']})
    for call in message.get('tool_calls') or []:
        function = call.get('function') or {}
        blocks.append({'type': 'tool_use', 'id': call.get('id') or '',
                       'name': function.get('name') or '',
                       'input': _input(function.get('arguments'))})
    return blocks


def convert(messages: list[dict], thinking: bool = True) -> tuple[str, list[dict]]:
    """Chat-completions messages as (system text, Messages API messages).

    Tool results that follow an assistant turn go into one user message, in
    the order of its tool calls; a call the engine did not run is answered
    with an error, so that every tool_use has its result.
    """
    system, out = [], []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get('role')
        index += 1
        if role == 'system':
            system.append(message.get('content') or '')
        elif role == 'user':
            if (message.get('content') or '').strip():
                out.append({'role': 'user', 'content': message['content']})
        elif role == 'assistant':
            blocks = _assistant_blocks(message, thinking)
            if not blocks:
                continue
            out.append({'role': 'assistant', 'content': blocks})
            wanted = [_block_id(b) for b in blocks if _block_type(b) == 'tool_use']
            results = {}
            while index < len(messages) and messages[index].get('role') == 'tool':
                results[messages[index].get('tool_call_id')] = messages[index].get('content') or ''
                index += 1
            if wanted:
                out.append({'role': 'user', 'content': [
                    _result(call_id, results.get(call_id, NOT_RUN)) for call_id in wanted]})
    return '\n\n'.join(part for part in system if part), out


def _result(call_id, content: str) -> dict:
    block = {'type': 'tool_result', 'tool_use_id': call_id, 'content': content or '(empty)'}
    if content.startswith('Error:'):
        block['is_error'] = True
    return block


def _replays_thinking(messages: list[dict]) -> bool:
    return any(_block_type(block) in ('thinking', 'redacted_thinking')
               for message in messages for block in message.get('native') or [])


# -------------------------------------------------------------- requests --

def _params(messages, tools, on_think, final, thinking=True) -> dict:
    system, converted = convert(messages, thinking)
    params = {'model': model(), 'max_tokens': _number('CLAUDE_MAX_TOKENS', 16000),
              'messages': converted,
              # The system prompt and the tools repeat on every round of a turn.
              'cache_control': {'type': 'ephemeral'}}
    if system:
        params['system'] = system
    if tools:
        params['tools'] = [_tool(schema) for schema in tools]
        # The last round keeps the tool list (changing it would invalidate the
        # turn's thinking blocks) and forbids calling one instead.
        params['tool_choice'] = {'type': 'none' if final else 'auto'}
    if not thinking:
        params['thinking'] = {'type': 'disabled'}
    elif on_think:
        params['thinking'] = {'type': 'adaptive', 'display': 'summarized'}
    effort = settings.get('CLAUDE_EFFORT').lower()
    if effort in EFFORTS and (thinking or effort not in ('xhigh', 'max')):
        params['output_config'] = {'effort': effort}
    return params


def _explain(error) -> str:
    detail = str(getattr(error, 'message', '') or error)[:300]
    return 'Claude returned an error (%s): %s' % (getattr(error, 'status_code', '?'), detail)


async def stream_chat(http, messages: list[dict], tools: list[dict] | None, on_text,
                      temperature: float | None = None, on_think=None,
                      final: bool = False) -> dict:
    """One streamed response; the same contract as llm.stream_chat.

    `temperature` is accepted for that contract and not sent: current Claude
    models reject sampling parameters. The returned dict also carries
    `native`, the response's content blocks, for the engine to keep with the
    assistant message.
    """
    try:
        try:
            message = await _stream(http, _params(messages, tools, on_think, final),
                                    on_text, on_think)
        except anthropic.BadRequestError:
            if not _replays_thinking(messages):
                raise
            # Something before a thinking block changed (the prompt was
            # shortened to fit): go on without the turn's thinking.
            message = await _stream(
                http, _params(messages, tools, None, final, thinking=False), on_text, None)
    except anthropic.AuthenticationError:
        raise LLMError('Claude refused the API key: check %s.'
                       % settings.variable('CLAUDE_API_KEY'))
    except anthropic.PermissionDeniedError as error:
        raise LLMError('This API key may not use %s (%s).' % (model(), _explain(error)))
    except anthropic.NotFoundError:
        raise LLMError('Claude does not know the model %s: check %s.'
                       % (model(), settings.variable('CLAUDE_MODEL')))
    except anthropic.RateLimitError:
        raise LLMError('Claude is rate limiting this server; try again in a minute.')
    except anthropic.APIStatusError as error:
        raise LLMError(_explain(error))
    except anthropic.APITimeoutError:
        raise LLMError('Claude did not answer in time.')
    except anthropic.APIConnectionError:
        raise LLMError('Cannot reach the Claude API from this server.')

    content = ''.join(block.text for block in message.content if block.type == 'text')
    calls = [{'id': block.id, 'type': 'function',
              'function': {'name': block.name, 'arguments': json.dumps(block.input)}}
             for block in message.content if block.type == 'tool_use']
    if message.stop_reason == 'refusal':
        await on_text(REFUSED)
        return {'content': content + REFUSED, 'tool_calls': [], 'finish_reason': 'refusal'}
    if message.stop_reason == 'max_tokens':
        # A tool call cut off mid-input would run with partial arguments.
        await on_text(CUT_OFF)
        return {'content': content + CUT_OFF, 'tool_calls': [], 'finish_reason': 'length'}
    return {'content': content, 'tool_calls': calls, 'finish_reason': message.stop_reason,
            'native': list(message.content) if calls else None}


async def _stream(http, params: dict, on_text, on_think):
    async with http.messages.stream(**params) as stream:
        async for event in stream:
            if event.type != 'content_block_delta':
                continue
            if event.delta.type == 'text_delta':
                await on_text(event.delta.text)
            elif event.delta.type == 'thinking_delta' and on_think and event.delta.thinking:
                await on_think(event.delta.thinking)
        return await stream.get_final_message()
