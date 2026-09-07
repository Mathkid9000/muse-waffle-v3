"""POST /api/edit/instr — add a new instrument (staff) to the score."""

from __future__ import annotations
import asyncio
import logging
import traceback
from api.schemas import EditInstrRequest, XmlResponse
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

# Display names + GM MIDI programs, indexed to match INSTR_NAMES in ml/api.py.
_INSTR_DISPLAY = [
    "Part", "Piano", "Harpsichord", "Organ", "Accordion",
    "Guitar", "Bass Guitar", "Violin", "Viola", "Cello",
    "Double Bass", "Harp", "Flute", "Oboe", "Clarinet",
    "Bassoon", "Saxophone", "Trumpet", "Horn", "Trombone",
    "Tuba", "Percussion", "Timpani", "Celesta",
    "Soprano", "Mezzo-Soprano", "Alto", "Tenor",
    "Baritone", "Bass", "Voice", "Choir",
]
_INSTR_MIDI = [
    0, 0, 6, 19, 21, 25, 33, 40, 41, 42, 43, 46, 73, 68, 71, 70,
    65, 56, 60, 57, 58, 0, 47, 8, 52, 52, 52, 52, 52, 52, 52, 52,
]


def _make_clef(idx: int):
    import music21.clef as clef

    return [
        clef.TrebleClef, clef.BassClef, clef.AltoClef, clef.TenorClef,
        clef.Treble8vbClef, clef.Treble8vaClef, clef.Bass8vbClef,
    ][idx]()


def _run_edit(body: EditInstrRequest) -> str:
    import music21 as m21

    orig_score = m21.converter.parseData(body.xml, format="musicxml")
    if not orig_score.parts:
        raise RuntimeError("Add instrument failed: score has no existing parts to align to")

    # Mirror the first staff's measure layout so the new staff stays aligned.
    ref = orig_score.parts[0]
    ref_measures = list(ref.getElementsByClass(m21.stream.Measure))
    if not ref_measures:
        raise RuntimeError("Add instrument failed: reference staff has no measures")

    new_part = m21.stream.Part()
    for ref_m in ref_measures:
        nm = m21.stream.Measure(number=ref_m.measureNumber)
        for ts in ref_m.getElementsByClass(m21.meter.TimeSignature):
            nm.insert(0, m21.meter.TimeSignature(ts.ratioString))
        nm.insert(0, m21.note.Rest(quarterLength=ref_m.barDuration.quarterLength))
        new_part.append(nm)

    first = list(new_part.getElementsByClass(m21.stream.Measure))[0]
    for ks in ref_measures[0].getElementsByClass(m21.key.KeySignature):
        first.insert(0, m21.key.KeySignature(ks.sharps))
    first.insert(0, _make_clef(body.new_clef))

    instrument = m21.instrument.Instrument(instrumentName=_INSTR_DISPLAY[body.instr])
    instrument.midiProgram = _INSTR_MIDI[body.instr]
    first.insert(0, instrument)

    orig_score.append(new_part)

    from notation import finish_edited
    return finish_edited(orig_score)


@router.post("/edit/instr", response_model=XmlResponse)
async def edit_instr(body: EditInstrRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_edit, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
