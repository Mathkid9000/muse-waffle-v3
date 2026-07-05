"""POST /api/edit/key_sig — change the key signature at a measure (all staves)."""

from __future__ import annotations
import asyncio
import logging
import traceback
from api.schemas import EditKeySigRequest, XmlResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_edit(body: EditKeySigRequest) -> str:
    import music21 as m21
    import music21.musicxml as musicxml
    from api.routers.infill import _ordinal_to_measure_numbers

    orig_score = m21.converter.parseData(body.xml, format="musicxml")
    if not orig_score.parts:
        raise RuntimeError("Key signature edit failed: score has no parts")

    # Normalize the tonic to music21 spelling ("Bb" -> "B-", "F#" -> "F#") and let
    # music21 derive the sharp/flat count for the requested mode (major/minor/modal).
    tonic = body.new_tonic.strip()
    tonic = tonic[:1].upper() + tonic[1:].replace("b", "-").replace("♭", "-").replace("♯", "#")
    mode = (body.new_mode or "major").strip().lower()
    try:
        new_key = m21.key.Key(tonic, mode)
    except Exception as e:
        raise RuntimeError(f"Key signature edit failed: bad key {body.new_tonic!r} {body.new_mode!r} ({e})")

    measure_num = _ordinal_to_measure_numbers(orig_score, body.measure_num, body.measure_num)[0]

    found = False
    for part in orig_score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            if measure.measureNumber == measure_num:
                measure.keySignature = m21.key.KeySignature(new_key.sharps)
                found = True
                break
    if not found:
        raise RuntimeError(f"Key signature edit failed: no measure {body.measure_num}")

    exporter = musicxml.m21ToXml.GeneralObjectExporter(orig_score)
    return exporter.parse().decode("utf-8")


@router.post("/edit/key_sig", response_model=XmlResponse)
async def edit_key_sig(body: EditKeySigRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_edit, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
