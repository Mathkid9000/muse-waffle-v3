"""POST /api/append — generate new measures at the end of an existing score."""

from __future__ import annotations
import asyncio
import json
import logging
import threading
import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.schemas import AppendRequest, XmlResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _stream(body: AppendRequest):
    from ml.api import append_stream

    return append_stream(
        xml_str=body.xml,
        num_bars=body.num_bars,
        max_new=body.max_new,
        temperature=body.temperature,
        top_k=body.top_k,
        device=body.device,
        model_name=body.model_name,
        log=logger.info,
    )


def _run_append(body: AppendRequest) -> str:
    # the one-shot endpoint is the streaming one drained; the last snapshot is the finished score
    xml = ""
    for _, xml, _ in _stream(body):
        pass
    if not xml:
        raise RuntimeError("Append generation failed: the model produced no complete bar")
    return xml


@router.post("/append", response_model=XmlResponse)
async def append(body: AppendRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_append, body)
    except Exception as e:
        logger.error("Append error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)


@router.post("/append/stream")
async def append_stream(body: AppendRequest):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for bar, xml, total in _stream(body):
                asyncio.run_coroutine_threadsafe(q.put(("data", bar, xml, total)), loop).result()
            asyncio.run_coroutine_threadsafe(q.put(("done",)), loop).result()
        except Exception as e:
            logger.error("Append stream error:\n%s", traceback.format_exc())
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
                _, bar, xml, total = item
                yield f"data: {json.dumps({'xml': xml, 'bar': bar, 'total': total})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
