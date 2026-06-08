"""POST /api/generate — create a new score from scratch."""

from __future__ import annotations
import asyncio
import logging
import traceback
from fastapi import APIRouter, HTTPException
from api.schemas import GenerateRequest, XmlResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _run_generate(body: GenerateRequest) -> str:
    from ml.tokenizer import BAR, KEY_OFF, TIME_OFF, TEMPO_OFF, INSTR_OFF
    from ml.generation_generator import generate_and_export

    header_tokens = [INSTR_OFF + i for i in body.instruments]
    first_bar_prefix = [BAR, KEY_OFF + body.key, TIME_OFF + body.time_sig, TEMPO_OFF + body.tempo]

    logs: list[str] = []
    result = generate_and_export(
        header_tokens=header_tokens,
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
