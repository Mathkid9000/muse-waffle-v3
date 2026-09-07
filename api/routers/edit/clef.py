"""POST /api/edit/clef — update the clef of a staff within a score."""

from __future__ import annotations
import asyncio
import json
import logging
import threading
import traceback
from api.schemas import EditClefRequest, XmlResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

def _run_edit(body: EditClefRequest) -> str:
    import music21 as m21
    import music21.clef as clef
    from api.routers.infill import _ordinal_to_measure_numbers

    logs: list[str] = []
    orig_score = m21.converter.parseData(body.xml, format="musicxml")
    if not orig_score.parts or len(orig_score.parts) <= body.staff_num:
        raise RuntimeError(f"Clef edit failed: | {body.staff_num} of {len(orig_score.parts)}" + " | ".join(logs[-5:]))
    
    measure_num = _ordinal_to_measure_numbers(orig_score, body.measure_num, body.measure_num)[0]
    measure = [ m for m in orig_score.parts[body.staff_num].getElementsByClass(m21.stream.Measure) if m.measureNumber == measure_num ][0]
    if not measure:
        raise RuntimeError(f"Clef edit failed: | No measure found ({body.measure_num})" + " | ".join(logs[-5:]))

    match body.new_clef:
        case 0:
            measure.clef = clef.TrebleClef()
        case 1:
            measure.clef = clef.BassClef()
        case 2:
            measure.clef = clef.AltoClef()
        case 3:
            measure.clef = clef.TenorClef()
        case 4:
            measure.clef = clef.Treble8vbClef()
        case 5:
            measure.clef = clef.Treble8vaClef()
        case 6: 
            measure.clef = clef.Bass8vbClef()

    from notation import finish_edited
    return finish_edited(orig_score)


@router.post("/edit/clef", response_model=XmlResponse)
async def append(body: EditClefRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_edit, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)