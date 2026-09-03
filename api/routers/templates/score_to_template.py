"""POST /api/templates/score-to-template — convert score to template."""

from __future__ import annotations
import asyncio
import logging
import re
import traceback
from functools import lru_cache
from api.schemas import XmlResponse
from fastapi import APIRouter, HTTPException
from api.routers.edit.instr import _INSTR_DISPLAY
from api.routers.templates.templateSchemas import (AnyTemplateItem, ChordTemplateItem, ClefTemplateItem,
                                                   InstrumentTemplateItem, KeyTemplateItem, NoteTemplateItem,
                                                   TempoTemplateItem, TemplateResponse, TimeTemplateItem)

logger = logging.getLogger(__name__)
router = APIRouter()

# The template's instrument vocabulary, indexed to match _INSTR_DISPLAY (and so INSTR_NAMES in ml/api.py).
_TEMPLATE_INSTRUMENTS = (
    "other", "piano", "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin",
    "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone",
    "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano",
    "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir",
)

# the tokenizer's chord qualities as the template spells them; sus4 has no template spelling of its
# own and keeps its suspension under 7sus rather than being dropped
_QUALITIES = {
    "": "maj", "m": "min", "m7": "min7", "7": "dom7", "o7": "dim7", "ø7": "m7b5", "dim": "dim",
    "7sus": "7sus", "sus2": "sus2", "sus": "7sus", "7+": "7aug", "+": "aug", "maj7": "maj7",
}


@lru_cache(maxsize=1)
def _instruments() -> dict[str, str]:
    """Template instrument name per canonical instrument name. Both sides run through the same
    canonicalization the tokenizer names parts with, so a staff reaches its template name under
    whatever spelling the typesetter used. Names that canonicalize onto one another (the picker's
    guitar and bass_guitar are both an acoustic guitar to music21) keep the first, and "Part"
    is left out entirely -- it canonicalizes to a piano and would take piano's own name."""
    import music21 as m21
    from tokenizer import canonical_instrument

    names: dict[str, str] = {}
    for display, template in zip(_INSTR_DISPLAY[1:], _TEMPLATE_INSTRUMENTS[1:]):
        names.setdefault(canonical_instrument(m21.instrument.Instrument(instrumentName=display)), template)
    return names


def _as_key(key_sig):
    """A key signature as a key. Key.asKey() re-reads an already-known key as major, so a key that
    knows its own mode is kept as it stands."""
    import music21 as m21

    return key_sig if isinstance(key_sig, m21.key.Key) else key_sig.asKey()


def _chord_item(chord_sym, num: int, pos: float) -> ChordTemplateItem:
    """A chord symbol as its degree in the key in effect, plus the quality it is written with."""
    import music21 as m21

    key_ctx = _as_key(chord_sym.getContextByClass(m21.key.KeySignature) or m21.key.KeySignature(0))
    rn = m21.roman.romanNumeralFromChord(m21.chord.Chord(chord_sym.pitches), key_ctx)
    quality = chord_sym.chordKindStr or re.split(r"/|add|omit|dim5", chord_sym.figure[len(chord_sym.root().name):])[0]
    return ChordTemplateItem(measure=num, pos=pos, degree=rn.romanNumeral,
                             rootNumeral=str(rn.scaleDegree), quality=_QUALITIES.get(quality, "maj"))


def _run_conversion(body: XmlResponse) -> list[AnyTemplateItem]:
    import music21 as m21
    from tokenizer import canonical_instrument

    orig_score = m21.converter.parseData(body.xml, format="musicxml")
    if not orig_score.parts:
        raise RuntimeError("Template conversion failed: score has no parts")

    template: list[AnyTemplateItem] = [
        InstrumentTemplateItem(measure=1, pos=0.0, voice_idx=idx,
                               instrument=_instruments().get(canonical_instrument(part.getInstrument()), "other"))
        for idx, part in enumerate(orig_score.parts)
    ]

    for voice_idx, part in enumerate(orig_score.parts):
        # key, time, tempo and chords belong to the score rather than to any one staff, so they are
        # read off the first part only -- every other part carries the same copy
        score_wide = voice_idx == 0
        # measures are numbered by position, the way the frontend counts them: a pickup is measure 1
        for num, measure in enumerate(part.getElementsByClass(m21.stream.Measure), 1):
            for el in measure.recurse():
                pos = float(el.getOffsetInHierarchy(measure))
                if isinstance(el, m21.clef.Clef):
                    template.append(ClefTemplateItem(measure=num, pos=pos, name=el.name, voice_idx=voice_idx))
                elif isinstance(el, m21.harmony.ChordSymbol):
                    if score_wide:
                        template.append(_chord_item(el, num, pos))
                elif isinstance(el, m21.note.NotRest):
                    # a chord is its notes: one item per pitch, all sharing the chord's position
                    template.extend(NoteTemplateItem(measure=num, pos=pos, voice_idx=voice_idx,
                                                     duration=float(el.duration.quarterLength), pitch=p.midi)
                                    for p in el.pitches)
                elif score_wide and isinstance(el, m21.key.KeySignature):
                    key = _as_key(el)
                    template.append(KeyTemplateItem(measure=num, pos=pos, key_tonic=key.tonic.name, key_mode=key.mode))
                elif score_wide and isinstance(el, m21.meter.TimeSignature):
                    template.append(TimeTemplateItem(measure=num, pos=pos, numerator=str(el.numerator),
                                                     denominator=str(el.denominator)))
                elif score_wide and isinstance(el, m21.tempo.MetronomeMark) and el.number is not None:
                    template.append(TempoTemplateItem(measure=num, pos=pos, tempo=round(el.number)))

    template.sort(key=lambda item: (item.measure, item.pos))
    return template


@router.post("/templates/score-to-template", response_model=TemplateResponse)
async def convert(body: XmlResponse):
    loop = asyncio.get_running_loop()
    try:
        template = await loop.run_in_executor(None, _run_conversion, body)
    except Exception as e:
        logger.error("Edit error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return TemplateResponse(template=template)
