"""The Claude backend (claude.py): message conversion, one streamed response,
and a turn through the engine and the HTTP route. The API is a scripted fake
client; nothing here reaches Anthropic."""
import asyncio
import json
from types import SimpleNamespace as NS

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

anthropic = pytest.importorskip('anthropic')

from proposer import claude, llm, server  # noqa: E402
import dataset_pack  # noqa: E402

TOOLS = [{'type': 'function', 'function': {
    'name': 'read_paper', 'description': 'Read a paper',
    'parameters': {'type': 'object', 'properties': {'doi': {'type': 'string'}}}}}]


def text(value):
    return NS(type='text', text=value)


def thinking(value='', signature='sig'):
    return NS(type='thinking', thinking=value, signature=signature)


def tool_use(name, arguments, call_id='toolu_1'):
    return NS(type='tool_use', id=call_id, name=name, input=arguments)


def reply(*blocks, stop='end_turn'):
    return NS(content=list(blocks), stop_reason=stop)


class FakeStream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for block in self.message.content:
            if block.type == 'text':
                for piece in (block.text[:3], block.text[3:]):
                    if piece:
                        yield NS(type='content_block_delta',
                                 delta=NS(type='text_delta', text=piece))
            elif block.type == 'thinking':
                yield NS(type='content_block_delta',
                         delta=NS(type='thinking_delta', thinking=block.thinking))
        yield NS(type='message_stop')

    async def get_final_message(self):
        return self.message


class FakeClient:
    """Answers with the scripted replies in turn; an exception is raised."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.messages = NS(stream=self.stream)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, **params):
        self.requests.append(params)
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return FakeStream(answer)


def api_error(kind, status):
    request = httpx.Request('POST', 'https://api.anthropic.com/v1/messages')
    return kind('no', response=httpx.Response(status, request=request), body=None)


def chat(client, messages, tools=TOOLS, think=False, final=False):
    said, thought = [], []

    async def on_text(delta):
        said.append(delta)

    async def on_think(delta):
        thought.append(delta)

    result = asyncio.run(claude.stream_chat(
        client, messages, tools, on_text, on_think=on_think if think else None, final=final))
    return result, ''.join(said), ''.join(thought)


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv('PROPOSER_CLAUDE_API_KEY', 'sk-test')
    for name in ('MODEL', 'EFFORT', 'MAX_TOKENS', 'CONTEXT_CHARS'):
        monkeypatch.delenv('PROPOSER_CLAUDE_' + name, raising=False)


# ----------------------------------------------------------- settings --

def test_configured_by_its_own_key_or_the_usual_one(monkeypatch):
    assert claude.configured() and claude.model() == 'claude-haiku-5-5'
    monkeypatch.delenv('PROPOSER_CLAUDE_API_KEY')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    assert not claude.configured()
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-usual')
    assert claude.configured() and claude.api_key() == 'sk-usual'


# --------------------------------------------------------- conversion --

def test_convert_groups_tool_results_under_their_turn():
    system, messages = claude.convert([
        {'role': 'system', 'content': 'Rules.'},
        {'role': 'user', 'content': 'Replicate the paper'},
        {'role': 'assistant', 'content': 'Reading.', 'tool_calls': [
            {'id': 'a', 'function': {'name': 'read_paper', 'arguments': '{"doi": "10.1/x"}'}},
            {'id': 'b', 'function': {'name': 'read_paper', 'arguments': 'not json'}}]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': 'Error: bad arguments'},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'The paper text'},
        {'role': 'user', 'content': '[Server note] Answer now.'},
    ])
    assert system == 'Rules.'
    assert [m['role'] for m in messages] == ['user', 'assistant', 'user', 'user']
    assert messages[1]['content'] == [
        {'type': 'text', 'text': 'Reading.'},
        {'type': 'tool_use', 'id': 'a', 'name': 'read_paper', 'input': {'doi': '10.1/x'}},
        {'type': 'tool_use', 'id': 'b', 'name': 'read_paper', 'input': {}}]
    assert messages[2]['content'] == [
        {'type': 'tool_result', 'tool_use_id': 'a', 'content': 'The paper text'},
        {'type': 'tool_result', 'tool_use_id': 'b', 'content': 'Error: bad arguments',
         'is_error': True}]


def test_convert_sends_native_blocks_back_and_answers_every_call():
    native = [thinking(), tool_use('read_paper', {}, 'a'), tool_use('read_paper', {}, 'b')]
    conversation = [
        {'role': 'user', 'content': 'Go'},
        # The engine ran only the first call of the round.
        {'role': 'assistant', 'content': '', 'native': native,
         'tool_calls': [{'id': 'a', 'function': {'name': 'read_paper', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'text'}]
    _, messages = claude.convert(conversation)
    assert messages[1]['content'] == native
    assert [r['tool_use_id'] for r in messages[2]['content']] == ['a', 'b']
    assert messages[2]['content'][1] == {
        'type': 'tool_result', 'tool_use_id': 'b', 'content': claude.NOT_RUN, 'is_error': True}
    _, without = claude.convert(conversation, thinking=False)
    assert [b.type for b in without[1]['content']] == ['tool_use', 'tool_use']


def test_convert_skips_empty_messages():
    _, messages = claude.convert([{'role': 'user', 'content': 'Hi'},
                                  {'role': 'assistant', 'content': ''},
                                  {'role': 'user', 'content': 'Still there?'}])
    assert [m['content'] for m in messages] == ['Hi', 'Still there?']


# ------------------------------------------------------------ requests --

def test_text_is_streamed_and_no_sampling_parameters_are_sent():
    client = FakeClient(reply(thinking(), text('Three accessions.')))
    result, said, thought = chat(client, [{'role': 'system', 'content': 'Rules.'},
                                          {'role': 'user', 'content': 'How many?'}])
    assert said == 'Three accessions.' and thought == ''
    assert result == {'content': 'Three accessions.', 'tool_calls': [],
                      'finish_reason': 'end_turn', 'native': None}
    (request,) = client.requests
    assert request['model'] == 'claude-haiku-5-5' and request['system'] == 'Rules.'
    assert request['tools'][0] == {'name': 'read_paper', 'description': 'Read a paper',
                                   'input_schema': TOOLS[0]['function']['parameters']}
    assert request['tool_choice'] == {'type': 'auto'}
    assert request['cache_control'] == {'type': 'ephemeral'}
    assert not {'temperature', 'top_p', 'thinking', 'output_config'} & set(request)


def test_reasoning_is_shown_only_when_asked_for(monkeypatch):
    monkeypatch.setenv('PROPOSER_CLAUDE_EFFORT', 'low')
    client = FakeClient(reply(thinking('Count the rows.'), text('Three.')))
    _, said, thought = chat(client, [{'role': 'user', 'content': 'How many?'}], think=True)
    assert (said, thought) == ('Three.', 'Count the rows.')
    assert client.requests[0]['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
    assert client.requests[0]['output_config'] == {'effort': 'low'}


def test_tool_calls_come_back_with_their_native_blocks():
    blocks = [thinking(), text('Reading.'), tool_use('read_paper', {'doi': '10.1/x'})]
    result, _, _ = chat(FakeClient(reply(*blocks, stop='tool_use')),
                        [{'role': 'user', 'content': 'Replicate 10.1/x'}])
    assert result['tool_calls'] == [{'id': 'toolu_1', 'type': 'function', 'function': {
        'name': 'read_paper', 'arguments': '{"doi": "10.1/x"}'}}]
    assert result['native'] == blocks


def test_the_last_round_keeps_the_tools_and_forbids_calling_them():
    client = FakeClient(reply(text('Done.')))
    chat(client, [{'role': 'user', 'content': 'Go'}], final=True)
    assert client.requests[0]['tools'] and client.requests[0]['tool_choice'] == {'type': 'none'}


def test_a_rejected_replay_of_thinking_is_retried_without_it():
    conversation = [
        {'role': 'user', 'content': 'Go'},
        {'role': 'assistant', 'content': '', 'native': [thinking(), tool_use('read_paper', {})],
         'tool_calls': [{'id': 'toolu_1', 'function': {'name': 'read_paper', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'shortened'}]
    client = FakeClient(api_error(anthropic.BadRequestError, 400), reply(text('Here.')))
    result, said, _ = chat(client, conversation, think=True)
    assert said == 'Here.' and result['content'] == 'Here.'
    first, second = client.requests
    assert [b.type for b in first['messages'][1]['content']] == ['thinking', 'tool_use']
    assert [b.type for b in second['messages'][1]['content']] == ['tool_use']
    assert second['thinking'] == {'type': 'disabled'}


def test_a_bad_request_without_thinking_is_reported():
    client = FakeClient(api_error(anthropic.BadRequestError, 400))
    with pytest.raises(llm.LLMError, match=r'Claude returned an error \(400\)'):
        chat(client, [{'role': 'user', 'content': 'Go'}])
    assert len(client.requests) == 1


def test_refusals_and_cut_off_answers_run_no_tools():
    result, said, _ = chat(FakeClient(reply(stop='refusal')),
                           [{'role': 'user', 'content': 'Go'}])
    assert said == claude.REFUSED and result['tool_calls'] == []
    cut = reply(text('Partly'), tool_use('read_paper', {'doi': '10.'}), stop='max_tokens')
    result, said, _ = chat(FakeClient(cut), [{'role': 'user', 'content': 'Go'}])
    assert result['tool_calls'] == [] and said.endswith(claude.CUT_OFF)


@pytest.mark.parametrize('kind, status, message', [
    (anthropic.AuthenticationError, 401, 'refused the API key: check PROPOSER_CLAUDE_API_KEY'),
    (anthropic.NotFoundError, 404, 'does not know the model claude-haiku-5-5'),
    (anthropic.RateLimitError, 429, 'rate limiting'),
    (anthropic.InternalServerError, 529, r'returned an error \(529\)'),
])
def test_api_errors_become_messages_for_the_user(kind, status, message):
    with pytest.raises(llm.LLMError, match=message):
        chat(FakeClient(api_error(kind, status)), [{'role': 'user', 'content': 'Go'}])


def test_an_unreachable_api_is_said_plainly():
    request = httpx.Request('POST', 'https://api.anthropic.com/v1/messages')
    error = anthropic.APIConnectionError(request=request)
    with pytest.raises(llm.LLMError, match='Cannot reach the Claude API'):
        chat(FakeClient(error), [{'role': 'user', 'content': 'Go'}])


# ------------------------------------------------- through the engine --

def events_of(response):
    events = []
    for block in response.text.strip().split('\n\n'):
        lines = block.split('\n')
        if lines[0].startswith(':'):
            continue
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def test_a_turn_on_claude_through_the_route(monkeypatch):
    blocks = [thinking(), tool_use('no_such_tool', {})]
    client = FakeClient(reply(*blocks, stop='tool_use'), reply(text('Nothing to propose yet.')))
    monkeypatch.setattr(claude, 'client', lambda transport=None: client)
    app = FastAPI()
    app.include_router(server.router(dataset_pack.pack(), claude), prefix='/api')
    http = TestClient(app)

    assert http.get('/api/assistant/info').json() == {
        'pack': 'tersect', 'configured': True, 'model': 'claude-haiku-5-5'}
    response = http.post('/api/assistant', json={
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'context': {'draft_id': None, 'step': 'reference', 'form': None, 'build': None}})
    assert response.status_code == 200, response.text
    events = events_of(response)
    assert ('text', {'delta': 'Not'}) in events and events[-1] == ('done', {})

    first, second = client.requests
    assert first['system'] == second['system'] and first['tools'] == second['tools']
    assert second['messages'][-2]['content'] == blocks          # sent back as it came
    (result,) = second['messages'][-1]['content']
    assert result['tool_use_id'] == 'toolu_1' and result['is_error']
    assert 'unknown tool' in result['content']


def test_the_route_is_off_without_a_key(monkeypatch):
    monkeypatch.delenv('PROPOSER_CLAUDE_API_KEY')
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    app = FastAPI()
    app.include_router(server.router(dataset_pack.pack(), claude), prefix='/api')
    response = TestClient(app).post('/api/assistant', json={
        'messages': [{'role': 'user', 'content': 'Hello'}]})
    assert response.status_code == 503 and 'PROPOSER_CLAUDE_API_KEY' in response.json()['detail']
