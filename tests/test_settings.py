"""Settings under an embedding application's own variable names."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from proposer import engine, llm, net, server, settings
import dataset_pack


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ('URL', 'MODEL', 'TIMEOUT', 'CONTEXT_CHARS', 'ALLOW_PRIVATE_URLS'):
        monkeypatch.delenv('PROPOSER_' + name, raising=False)
    yield
    settings.reset()


def test_defaults_and_own_names(monkeypatch):
    assert not llm.configured() and llm.model() == 'qwen3:14b'
    assert engine.turn_timeout() == 600 and engine.context_chars() == 24000
    monkeypatch.setenv('PROPOSER_URL', 'http://a.test/v1/')
    monkeypatch.setenv('PROPOSER_CONTEXT_CHARS', 'many')
    assert llm.base_url() == 'http://a.test/v1' and engine.context_chars() == 24000
    assert settings.variable('URL') == 'PROPOSER_URL'


def test_aliases(monkeypatch):
    settings.alias('APP_ASSISTANT_', ALLOW_PRIVATE_URLS='APP_ALLOW_PRIVATE')
    monkeypatch.setenv('PROPOSER_MODEL', 'theirs')
    monkeypatch.setenv('APP_ASSISTANT_MODEL', 'ours')
    monkeypatch.setenv('APP_ASSISTANT_TIMEOUT', '30')
    monkeypatch.setenv('APP_ALLOW_PRIVATE', 'yes')
    assert llm.model() == 'ours' and engine.turn_timeout() == 30    # the alias wins
    assert net.private_urls_allowed()
    monkeypatch.delenv('APP_ASSISTANT_MODEL')
    assert llm.model() == 'theirs'
    assert settings.variables('ALLOW_PRIVATE_URLS') == [
        'APP_ALLOW_PRIVATE', 'APP_ASSISTANT_ALLOW_PRIVATE_URLS', 'PROPOSER_ALLOW_PRIVATE_URLS']
    # Messages name the variable the application's operators know.
    app = FastAPI()
    app.include_router(server.router(dataset_pack.pack()), prefix='/api')
    response = TestClient(app).post('/api/assistant', json={
        'messages': [{'role': 'user', 'content': 'hi'}]})
    assert response.status_code == 503 and 'APP_ASSISTANT_URL' in response.json()['detail']
