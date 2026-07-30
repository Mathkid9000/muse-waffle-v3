"""POST /api/generate — create a new score from scratch."""

from __future__ import annotations
import asyncio
import json
import logging
import threading
import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.schemas import GenerateRequest, XmlResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _stream(body: GenerateRequest):
    from ml.api import generate_stream

    return generate_stream(
        instruments=body.instruments,
        key=body.key,
        time_sig=body.time_sig,
        tempo=body.tempo,
        num_bars=body.num_bars,
        max_new=body.max_new,
        temperature=body.temperature,
        top_k=body.top_k,
        device=body.device,
        model_name=body.model_name,
        log=logger.info,
    )


def _run_generate(body: GenerateRequest) -> str:
    # the one-shot endpoint is the streaming one drained: the model has a single way to build a
    # score, and the last snapshot is the finished one
    xml = ""
    for _, xml, _, _ in _stream(body):
        pass
    if not xml:
        raise RuntimeError("Generation failed: the model produced no complete bar")
    return xml


@router.post("/generate", response_model=XmlResponse)
async def generate(body: GenerateRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_generate, body)
    except Exception as e:
        logger.error("Generate error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)


@router.post("/generate/stream")
async def generate_stream(body: GenerateRequest):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for bar, xml, total, num_instr in _stream(body):
                asyncio.run_coroutine_threadsafe(q.put(("data", bar, xml, total, num_instr)), loop).result()
            asyncio.run_coroutine_threadsafe(q.put(("done",)), loop).result()
        except Exception as e:
            logger.error("Generate stream error:\n%s", traceback.format_exc())
            asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop).result()

    threading.Thread(target=run, daemon=True).start()

    async def event_gen():
        while True:
            item = await q.get()
            if item[0] == "done":
                yield 'data: {"done":true}\n\n'
                break
            elif item[0] == "error":
                yield f"data: {json.dumps({'error': item[1]})}\n\n"
                break
            else:
                _, bar, xml, total, instr_num = item
                yield f"data: {json.dumps({'xml': xml, 'bar': bar, 'total': total, 'instr_num': instr_num})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
