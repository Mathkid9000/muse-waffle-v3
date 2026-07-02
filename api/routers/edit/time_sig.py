"""POST /api/edit/time_sig — change the time signature at a measure (all staves)."""

from __future__ import annotations
import asyncio
import logging
import traceback
from api.schemas import EditTimeSigRequest, XmlResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_edit(body: EditTimeSigRequest) -> str:
    import music21 as m21
    import music21.musicxml as musicxml
    from api.routers.infill import _ordinal_to_measure_numbers
    from ml.tokenizer import TIME_SIGS

    orig_score = m21.converter.parseData(body.xml, format="musicxml")
    if not orig_score.parts:
        raise RuntimeError("Time signature edit failed: score has no parts")

    # A time signature is global; apply it to the same measure in every staff.
    measure_num = _ordinal_to_measure_numbers(orig_score, body.measure_num, body.measure_num)[0]

    found = False
    for part in orig_score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            if measure.measureNumber == measure_num:
                measure.timeSignature = m21.meter.TimeSignature(TIME_SIGS[body.new_time])
                found = True
                break
    if not found:
        raise RuntimeError(f"Time signature edit failed: no measure {body.measure_num}")

    exporter = musicxml.m21ToXml.GeneralObjectExporter(orig_score)
    return exporter.parse().decode("utf-8")


@router.post("/edit/time_sig", response_model=XmlResponse)
async def edit_time_sig(body: EditTimeSigRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_edit, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
