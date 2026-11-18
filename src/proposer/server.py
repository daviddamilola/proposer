"""HTTP interface: an assistant for one pack as a FastAPI router.

    POST /assistant         Request -> text/event-stream
    GET  /assistant/info    {pack, configured, model}

Events:

    event: status    data: {"text": "Reading the paper…"}
    event: text      data: {"delta": "…"}
    event: thinking  data: {"delta": "…"}
    event: proposal  data: Proposal
    event: done      data: {}
    event: error     data: {"message": "…"}     (then the stream ends)

Mount it in an existing app (app.include_router(router(pack), prefix='/api'))
or run it alone with create_app(pack).
"""
import asyncio
import json
import logging

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import engine, llm, settings
from .pack import Pack

log = logging.getLogger('proposer')

KEEPALIVE = 15             # seconds between SSE comments while waiting


def sse(event: str, data) -> str:
    return 'event: %s\ndata: %s\n\n' % (event, json.dumps(data, separators=(',', ':')))


async def event_stream(pack: Pack, request: engine.Request):
    queue: asyncio.Queue = asyncio.Queue()
    END = object()

    async def emit(event, data):
        await queue.put(sse(event, data))

    async def produce():
        try:
            async with asyncio.timeout(engine.turn_timeout()):
                await engine.run_turn(pack, request, emit)
            await emit('done', {})
        except (llm.LLMError, engine.BadRequest) as error:
            await emit('error', {'message': str(error)})
        except TimeoutError:
            await emit('error', {'message': 'The assistant took longer than %d seconds; '
                                 'try a narrower question.' % engine.turn_timeout()})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.exception('Assistant turn failed')
            await emit('error', {'message': 'The assistant failed: %s'
                                 % (str(error) or type(error).__name__)})
        finally:
            await queue.put(END)

    task = asyncio.create_task(produce())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), KEEPALIVE)
            except TimeoutError:
                yield ': keepalive\n\n'
                continue
            if item is END:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()   # the client went away


def router(pack: Pack) -> APIRouter:
    api = APIRouter()

    @api.get('/assistant/info')
    def info():
        return {'pack': pack.name, 'configured': llm.configured(),
                'model': llm.model() if llm.configured() else None}

    @api.post('/assistant')
    async def post_assistant(body: engine.Request):
        if not llm.configured():
            raise HTTPException(503, 'The assistant is not configured on this server '
                                     '(set %s to an OpenAI-compatible endpoint, '
                                     'e.g. a local Ollama).' % settings.variable('URL'))
        try:
            engine.history(body)   # 400 before the stream starts
        except engine.BadRequest as error:
            raise HTTPException(400, str(error))
        return StreamingResponse(event_stream(pack, body), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache',
                                          'X-Accel-Buffering': 'no'})

    return api


def create_app(pack: Pack) -> FastAPI:
    app = FastAPI(title='proposer: %s' % pack.name)
    app.include_router(router(pack), prefix='/api')
    return app
