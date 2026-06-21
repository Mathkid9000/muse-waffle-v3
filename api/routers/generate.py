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


def _run_generate(body: GenerateRequest) -> str:
    from ml.tokenizer import BAR, KEY_OFF, TIME_OFF, TEMPO_OFF
    from ml.generation_generator import generate_and_export

    first_bar_prefix = [BAR, KEY_OFF + body.key, TIME_OFF + body.time_sig, TEMPO_OFF + body.tempo]

    logs: list[str] = []
    result = generate_and_export(
        instr_indices=body.instruments,
        first_bar_prefix=first_bar_prefix,
        target_bars=body.num_bars,
        max_new=body.max_new,
        temperature=body.temperature,
        top_k=body.top_k,
        log=logs.append,
        device=body.device,
        model_name=body.model_name,
    )

    xml_path = result.get("xml", "")
    if not xml_path:
        raise RuntimeError("Generation failed: " + " | ".join(logs[-5:]))

    with open(xml_path, "r", encoding="utf-8") as f:
        return f.read()


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
    from ml.tokenizer import BAR, KEY_OFF, TIME_OFF, TEMPO_OFF
    from ml.generation_generator import generate_and_stream

    first_bar_prefix = [BAR, KEY_OFF + body.key, TIME_OFF + body.time_sig, TEMPO_OFF + body.tempo]

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for bar, xml, total, num_instr in generate_and_stream(
                instr_indices=body.instruments,
                first_bar_prefix=first_bar_prefix,
                target_bars=body.num_bars,
                max_new=body.max_new,
                temperature=body.temperature,
                top_k=body.top_k,
                device=body.device,
                model_name=body.model_name,
            ):
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
