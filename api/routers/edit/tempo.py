"""POST /api/edit/tempo — set the tempo at a measure (first staff carries it)."""

from __future__ import annotations
import asyncio
import logging
import traceback
from api.schemas import EditTempoRequest, XmlResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_edit(body: EditTempoRequest) -> str:
    import music21 as m21
    import music21.musicxml as musicxml
    from api.routers.infill import _ordinal_to_measure_numbers
    from ml.tokenizer import TEMPO_BUCKETS

    orig_score = m21.converter.parseData(body.xml, format="musicxml")
    if not orig_score.parts:
        raise RuntimeError("Tempo edit failed: score has no parts")

    # Bucket index matches GenerateRequest.tempo (0=slowest … 15=fastest); use the
    # bucket's lower-boundary BPM as the concrete metronome value.
    bpm = TEMPO_BUCKETS[body.new_tempo]
    measure_num = _ordinal_to_measure_numbers(orig_score, body.measure_num, body.measure_num)[0]

    # Tempo is a global direction; music21/MusicXML attach it to the first staff.
    part = orig_score.parts[0]
    target = [m for m in part.getElementsByClass(m21.stream.Measure) if m.measureNumber == measure_num]
    if not target:
        raise RuntimeError(f"Tempo edit failed: no measure {body.measure_num}")
    measure = target[0]

    for mm in list(measure.getElementsByClass(m21.tempo.MetronomeMark)):
        measure.remove(mm)
    measure.insert(0, m21.tempo.MetronomeMark(number=bpm))

    exporter = musicxml.m21ToXml.GeneralObjectExporter(orig_score)
    return exporter.parse().decode("utf-8")


@router.post("/edit/tempo", response_model=XmlResponse)
async def edit_tempo(body: EditTempoRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_edit, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
