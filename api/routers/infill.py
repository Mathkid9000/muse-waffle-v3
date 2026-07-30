"""POST /api/infill — fill empty measure ranges in an existing MusicXML score."""

from __future__ import annotations
import asyncio
import json
import logging
import threading
import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.schemas import InfillRequest, XmlResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _ordinal_to_measure_numbers(score, start: int, end: int) -> tuple[int, int]:
    """Translate 1-based visual/ordinal measure positions into music21 measure.number values.

    The frontend selects measures by their position in the rendered score: Verovio draws
    every measure — including a pickup/anacrusis — in document order, so the user's Nth
    selected measure is the Nth measure element. music21, however, numbers a pickup measure
    0 (implicit="yes"), so ``measure.number`` and the visual ordinal diverge once a pickup
    (or any measure-length anomaly that shifts numbering) is present. Map by position using
    the first part's measure list so the rest of the pipeline — which keys on measure.number —
    blanks and fills exactly the measures the user clicked.
    """
    import music21 as m21

    if not score.parts:
        return start, end
    measures = list(score.parts[0].getElementsByClass(m21.stream.Measure))
    if not measures:
        return start, end
    nums = [m.number if m.number is not None else 0 for m in measures]

    def at(ordinal: int) -> int:
        idx = min(max(ordinal - 1, 0), len(nums) - 1)
        return nums[idx]

    return at(start), at(end)


def _stream(body: InfillRequest):
    from ml.api import infill_stream

    # measures go over as the ordinals the frontend selected: the model reads the score as a
    # stream of bars, so the Nth rendered measure is the Nth bar of that stream and no translation
    # through music21's numbering (which _ordinal_to_measure_numbers does for the edit endpoints)
    # is needed here
    return infill_stream(
        xml_str=body.xml,
        start_measure=body.start_measure,
        end_measure=body.end_measure,
        parts=body.parts,
        temperature=body.temperature,
        top_k=body.top_k,
        device=body.device,
        model_name=body.model_name,
        log=logger.info,
    )


def _run_infill(body: InfillRequest) -> str:
    # the one-shot endpoint is the streaming one drained; the last snapshot is the filled score
    xml = ""
    for _, xml, _ in _stream(body):
        pass
    if not xml:
        raise RuntimeError("Infill failed: the model produced no complete bar")
    return xml


@router.post("/infill", response_model=XmlResponse)
async def infill(body: InfillRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_infill, body)
    except Exception as e:
        logger.error("Infill error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)


@router.post("/infill/stream")
async def infill_stream(body: InfillRequest):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for bar, xml, total in _stream(body):
                asyncio.run_coroutine_threadsafe(q.put(("data", bar, xml, total)), loop).result()
            asyncio.run_coroutine_threadsafe(q.put(("done",)), loop).result()
        except Exception as e:
            logger.error("Infill stream error:\n%s", traceback.format_exc())
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
