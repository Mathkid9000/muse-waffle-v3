"""POST /api/edit/rm-measure — remove a measure range from every staff."""

from __future__ import annotations
import asyncio
import logging
import traceback
from api.schemas import EditRmMeasureRequest, XmlResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_edit(body: EditRmMeasureRequest) -> str:
    import music21 as m21
    from api.routers.infill import _ordinal_to_measure_numbers
    from notation import carry_state, finish_edited

    orig_score = m21.converter.parseData(body.xml, format="musicxml")
    if not orig_score.parts:
        raise RuntimeError("Remove measures failed: score has no parts")

    start_m, end_m = _ordinal_to_measure_numbers(orig_score, body.measure_start_num, body.measure_end_num)
    if start_m > end_m:
        start_m, end_m = end_m, start_m

    removed_any = False
    for part in orig_score.parts:
        measures = list(part.getElementsByClass(m21.stream.Measure))
        # a removed bar takes the clef, key, meter and tempo it set with it, so the bar that
        # closes the gap is handed whatever it was reading from them
        cut = [i for i, measure in enumerate(measures) if start_m <= measure.measureNumber <= end_m]
        if cut: carry_state(measures, cut[-1] + 1)

        keep = []
        for measure in measures:
            if start_m <= measure.measureNumber <= end_m:
                part.remove(measure)
                removed_any = True
            else:
                keep.append(measure)

        # Repack the survivors: close the gap left by the removed measures and
        # renumber so measure numbers stay contiguous.
        offset = 0.0
        for i, measure in enumerate(keep, start=1):
            part.setElementOffset(measure, offset)
            measure.number = i
            offset += measure.barDuration.quarterLength

    if not removed_any:
        raise RuntimeError(
            f"Remove measures failed: no measures in range {body.measure_start_num}-{body.measure_end_num}"
        )

    return finish_edited(orig_score)


@router.post("/edit/rm-measure", response_model=XmlResponse)
async def edit_rm_measure(body: EditRmMeasureRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_edit, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
