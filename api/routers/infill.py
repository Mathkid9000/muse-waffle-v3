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


def _mark_measures_empty(score, start: int, end: int, parts: list[int] | None = None) -> None:
    """Replace note content in [start, end] with a hidden rest; preserve time/key/tempo.

    If *parts* is given, only those 0-indexed part indices are blanked; the rest keep
    their content so the model can use them as cross-staff context.
    """
    import music21 as m21

    _structural = (m21.key.KeySignature, m21.meter.TimeSignature, m21.tempo.MetronomeMark, m21.clef.Clef)
    for idx, part in enumerate(score.parts):
        if parts is not None and idx not in parts:
            continue
        for measure in part.getElementsByClass(m21.stream.Measure):
            m_num = measure.number if measure.number is not None else 0
            if start <= m_num <= end:
                keep = [el for el in measure if isinstance(el, _structural)]
                measure.elements = []
                rest = m21.note.Rest(quarterLength=4)
                rest.style.hideObjectOnPrint = True
                measure.insert(0, rest)
                for el in keep:
                    measure.insert(el.offset, el)


def _run_infill(body: InfillRequest) -> str:
    import music21 as m21
    from ml.infill_generator import infill_and_export

    score = m21.converter.parseData(body.xml, format="musicxml")
    start, end = _ordinal_to_measure_numbers(score, body.start_measure, body.end_measure)
    _mark_measures_empty(score, start, end, body.parts)

    logs: list[str] = []
    result = infill_and_export(
        score=score,
        model_name=body.model_name,
        max_new=500,
        temperature=body.temperature,
        top_k=body.top_k,
        log=logs.append,
        device=body.device,
        gaps=[(start, end)],
        target_parts=body.parts,
    )

    xml_path = result.get("xml", "")
    if not xml_path:
        raise RuntimeError("Infill failed: " + " | ".join(logs[-5:]))

    with open(xml_path, "r", encoding="utf-8") as f:
        return f.read()


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
    import music21 as m21
    from ml.infill_generator import infill_and_stream

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            score = m21.converter.parseData(body.xml, format="musicxml")
            start, end = _ordinal_to_measure_numbers(score, body.start_measure, body.end_measure)
            _mark_measures_empty(score, start, end, body.parts)
            for bar, xml, total in infill_and_stream(
                score=score,
                model_name=body.model_name,
                max_new=500,
                temperature=body.temperature,
                top_k=body.top_k,
                device=body.device,
                gaps=[(start, end)],
                target_parts=body.parts,
            ):
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
