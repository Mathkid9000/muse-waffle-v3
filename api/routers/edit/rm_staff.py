"""POST /api/edit/rm-staff — remove a whole staff (part) from the score."""

from __future__ import annotations
import asyncio
import logging
import traceback
from api.schemas import EditRmStaffRequest, XmlResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_edit(body: EditRmStaffRequest) -> str:
    import music21 as m21
    import music21.musicxml as musicxml

    orig_score = m21.converter.parseData(body.xml, format="musicxml")
    parts = list(orig_score.parts)
    if not parts:
        raise RuntimeError("Remove staff failed: score has no parts")
    if body.staff_num >= len(parts):
        raise RuntimeError(
            f"Remove staff failed: staff {body.staff_num} of {len(parts)}"
        )

    orig_score.remove(parts[body.staff_num], recurse=True)

    exporter = musicxml.m21ToXml.GeneralObjectExporter(orig_score)
    return exporter.parse().decode("utf-8")


@router.post("/edit/rm-staff", response_model=XmlResponse)
async def edit_rm_staff(body: EditRmStaffRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_edit, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
