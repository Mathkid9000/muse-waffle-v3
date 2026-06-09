"""POST /api/append — generate new measures at the end of an existing score."""

from __future__ import annotations
import asyncio
import logging
import traceback
from fastapi import APIRouter, HTTPException
from api.schemas import AppendRequest, XmlResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _run_append(body: AppendRequest) -> str:
    from ml.append_generator import append_and_export

    logs: list[str] = []
    result = append_and_export(
        xml_str=body.xml,
        num_bars=body.num_bars,
        max_new=body.max_new,
        temperature=body.temperature,
        top_k=body.top_k,
        log=logs.append,
        device=body.device,
        model_name=body.model_name,
    )

    xml_path = result.get("xml", "")
    if not xml_path:
        raise RuntimeError("Append generation failed: " + " | ".join(logs[-5:]))

    with open(xml_path, "r", encoding="utf-8") as f:
        return f.read()


@router.post("/append", response_model=XmlResponse)
async def append(body: AppendRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_append, body)
    except Exception as e:
        logger.error("Append error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
