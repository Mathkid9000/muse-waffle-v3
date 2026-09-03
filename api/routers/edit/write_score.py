"""POST /api/edit/write-score — write one score onto another, a copy and paste."""

from __future__ import annotations
import asyncio
import copy
import logging
import traceback
from api.schemas import EditWriteScoreRequest, XmlResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_edit(body: EditWriteScoreRequest) -> str:
    import music21 as m21
    import music21.musicxml as musicxml

    source = m21.converter.parseData(body.source_xml, format="musicxml")
    if not source.parts:
        raise RuntimeError("Write score failed: the score being copied has no parts")
    target = m21.converter.parseData(body.xml, format="musicxml") if body.xml else m21.stream.Score()

    # measures are addressed by position, the way the frontend counts them
    start = body.measure_start_num - 1
    for idx, part in enumerate(source.parts):
        if idx < len(target.parts):
            into = target.parts[idx]
        else:
            into = m21.stream.Part()
            into.insert(0, copy.deepcopy(part.getInstrument()))
            target.insert(0, into)

        measures = list(into.getElementsByClass(m21.stream.Measure))
        copied = [copy.deepcopy(m) for m in part.getElementsByClass(m21.stream.Measure)]
        # the copy replaces what it lands on; a paste past the end pads the gap with empty bars,
        # and whatever the target held on either side of the block is kept
        head = measures[:start] + [m21.stream.Measure() for _ in range(start - len(measures))]
        written = head + copied + measures[start + len(copied):]

        for measure in measures:
            into.remove(measure)
        offset = 0.0
        for num, measure in enumerate(written, start=1):
            measure.number = num
            into.insert(offset, measure)
            offset += measure.barDuration.quarterLength

    # a copy covering fewer staves than the target has leaves those staves short, and a score whose
    # parts run to different lengths renders ragged, so the short ones are padded with empty bars
    lengths = [len(list(part.getElementsByClass(m21.stream.Measure))) for part in target.parts]
    for part, length in zip(target.parts, lengths):
        offset = part.highestTime
        for num in range(length + 1, max(lengths) + 1):
            part.insert(offset, m21.stream.Measure(number=num))
            offset += part.getElementsByClass(m21.stream.Measure)[-1].barDuration.quarterLength

    exporter = musicxml.m21ToXml.GeneralObjectExporter(target)
    return exporter.parse().decode("utf-8")


@router.post("/edit/write-score", response_model=XmlResponse)
async def edit_write_score(body: EditWriteScoreRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_edit, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
