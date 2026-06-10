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


def _mark_measures_empty(score, start: int, end: int) -> None:
    """Replace note content in [start, end] with a hidden rest; preserve time/key/tempo."""
    import music21 as m21

    _structural = (m21.key.KeySignature, m21.meter.TimeSignature, m21.tempo.MetronomeMark, m21.clef.Clef)
    for part in score.parts:
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
    _mark_measures_empty(score, body.start_measure, body.end_measure)

    logs: list[str] = []
    result = infill_and_export(
        score=score,
        model_name=body.model_name,
        max_new=500,
        temperature=body.temperature,
        top_k=body.top_k,
        log=logs.append,
        device=body.device,
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
            _mark_measures_empty(score, body.start_measure, body.end_measure)
            for bar, xml, total in infill_and_stream(
                score=score,
                model_name=body.model_name,
                max_new=500,
                temperature=body.temperature,
                top_k=body.top_k,
                device=body.device,
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
