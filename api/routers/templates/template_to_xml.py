"""POST /api/templates/template-to-xml — render a template's filled-in fields as a score, for
previewing a template without asking the model to write its blanks."""

from __future__ import annotations
import asyncio
import logging
import traceback
from api.schemas import XmlResponse
from fastapi import APIRouter, HTTPException
from api.routers.templates.fill_template import _FIGURES
from api.routers.templates.templateSchemas import AnyTemplateItem, TemplateResponse
from music21 import stream
from music21 import note as note21
from music21 import chord as chord21
from music21 import duration as dur21
from music21 import pitch as pitch21
from music21 import instrument as instrument21
from music21 import clef as clef21
from music21 import key as key21
from music21 import meter as meter21
from music21 import tempo as tempo21
from music21 import harmony as harmony21
from music21 import roman as roman21
from music21 import expressions as expr21

logger = logging.getLogger(__name__)
router = APIRouter()


def _blank(kind: str) -> expr21.TextExpression:
    """The placeholder for wherever a template item has left its value(s) unset. There is no
    'blank clef' or 'blank key signature' in notation, so a still-empty field is named in text
    instead -- the same idea fill_template.py's canvas rows use a MASK token for."""
    text = expr21.TextExpression(f'{kind}?')
    text.style.fontStyle = 'italic'
    return text


def _run_conversion(template: list[AnyTemplateItem]) -> str:
    if not any(it.kind == "instrument" for it in template):
        raise RuntimeError("Template conversion failed: the template names no instrument to write for")

    score = stream.Score()
    parts: dict[int, stream.Part] = {}
    measures: dict[int, dict[int, stream.Measure]] = {}

    def measure_for(voice_idx: int, m_num: int) -> stream.Measure:
        voice_measures = measures[voice_idx]
        if m_num not in voice_measures:
            voice_measures[m_num] = stream.Measure()
        return voice_measures[m_num]

    for item in template:
        if item.kind == 'instrument':
            parts[item.voice_idx] = stream.Part()
            parts[item.voice_idx].append(instrument21.Instrument(instrumentName=item.instrument or None))
            if not item.instrument:
                parts[item.voice_idx].partName = 'instrument?'
            measures[item.voice_idx] = {}

    if not parts:
        raise RuntimeError("Template conversion failed: every instrument slot is still blank")
    top_voice = min(parts)   # tempo marks and chord symbols are written once, above the top staff

    # notes: several items sharing one (voice, measure, pos) slot are one chord's pitches, so they
    # are grouped here and turned into a single Chord rather than inserted as separate Notes
    # stacked on the same offset, which music21 reads back as a run of single notes instead of one
    # simultaneity. A slot with no pitches keeps the usual slashed, unpitched notehead; more than
    # one blank pitch in the same slot would otherwise collapse silently into that one notehead, so
    # its count is written out as a lyric instead. A blank duration gets the same treatment.
    slots: dict[tuple[int, int, float], list] = {}
    for item in template:
        if item.kind == 'note' and item.voice_idx in parts and item.measure is not None:
            slots.setdefault((item.voice_idx, item.measure, item.pos or 0.0), []).append(item)

    for (voice_idx, m_num, pos), items in slots.items():
        pitches = [it.pitch for it in items if it.pitch is not None]
        dur = next((it.duration for it in items if it.duration is not None), None)
        if not pitches:
            el = note21.Unpitched()
            el.notehead = 'slash'
            if len(items) > 1:
                el.lyric = f'×{len(items)}'
        elif len(pitches) == 1:
            el = note21.Note(pitch21.Pitch(midi=pitches[0]))
        else:
            el = chord21.Chord([pitch21.Pitch(midi=p) for p in pitches])
        el.duration = dur21.Duration(dur if dur is not None else 1.0)
        if dur is None:
            el.lyric = f'{el.lyric} dur?' if el.lyric else 'dur?'

        measure_for(voice_idx, m_num).insert(pos, el)

    # clefs: per-voice, either the named clef or a blank marker on that voice's own staff
    for item in template:
        if item.kind == 'clef' and item.voice_idx in parts and item.measure is not None:
            el = clef21.clefFromString(item.name) if item.name else _blank('clef')
            measure_for(item.voice_idx, item.measure).insert(item.pos or 0.0, el)

    # key, time, tempo and chords are score-wide rather than per-voice: key and time are written
    # into every staff, the way a real score repeats them on each one, while tempo marks and chord
    # symbols land once, above the top staff. Roman-numeral chords need the key in effect at that
    # point, so these are walked in score order, tracking the most recently placed key as we go.
    def _pos(it) -> float:
        return 0.0 if it.kind == 'masked' else (it.pos or 0.0)

    def _at(it):
        return it.measure_start if it.kind == 'masked' else it.measure

    score_wide = sorted(
        (it for it in template if it.kind in ('key', 'time', 'tempo', 'chord', 'masked') and _at(it) is not None),
        key=lambda it: (_at(it), _pos(it)),
    )
    current_key = key21.Key('C')
    for it in score_wide:
        if it.kind == 'key':
            if it.key_tonic and it.key_mode:
                current_key = key21.Key(it.key_tonic, it.key_mode)
                for voice_idx in parts:
                    measure_for(voice_idx, it.measure).insert(_pos(it), key21.Key(it.key_tonic, it.key_mode))
            else:
                for voice_idx in parts:
                    measure_for(voice_idx, it.measure).insert(_pos(it), _blank('key'))
        elif it.kind == 'time':
            if it.numerator and it.denominator:
                for voice_idx in parts:
                    measure_for(voice_idx, it.measure).insert(
                        _pos(it), meter21.TimeSignature(f'{it.numerator}/{it.denominator}'))
            else:
                for voice_idx in parts:
                    measure_for(voice_idx, it.measure).insert(_pos(it), _blank('time'))
        elif it.kind == 'tempo':
            el = tempo21.MetronomeMark(number=it.tempo) if it.tempo is not None else _blank('tempo')
            measure_for(top_voice, it.measure).insert(_pos(it), el)
        elif it.kind == 'chord':
            if it.degree and it.quality:
                root = roman21.RomanNumeral(it.degree, current_key).root().name
                el = harmony21.ChordSymbol(root + _FIGURES.get(it.quality, ''))
            else:
                el = _blank('chord')
            measure_for(top_voice, it.measure).insert(_pos(it), el)
        elif it.kind == 'masked':
            # the whole span is unwritten -- only the model can choose what actually goes there --
            # so it is left as empty bars in every voice, marked where the span begins
            for voice_idx in parts:
                for offset in range(it.mask_measure_length):
                    m = measure_for(voice_idx, it.measure_start + offset)
                    if offset == 0:
                        m.insert(0.0, _blank('masked'))

    # add all measures to parts, in measure order and padded with empty measures so every part's
    # bars line up even where a voice has nothing written in a given measure -- insert(m) alone
    # left every measure of a part at offset 0.0 (or, unsorted, out of order), which is invisible
    # with one voice but staggers every other voice against it once there is more than one
    all_measure_nums = sorted({n for voice_measures in measures.values() for n in voice_measures})
    for part_idx in parts:
        for m_num in all_measure_nums:
            m = measures[part_idx][m_num] if m_num in measures[part_idx] else stream.Measure()
            m.number = m_num
            parts[part_idx].append(m)
        score.insert(parts[part_idx])

    import music21
    exporter = music21.musicxml.m21ToXml.GeneralObjectExporter(score)
    xml_string = exporter.parse().decode('utf-8')

    if xml_string is None:
        raise RuntimeError("Template conversion failed: nothing in the template renders yet")
    return xml_string


@router.post("/templates/template-to-xml", response_model=XmlResponse)
async def convert(body: TemplateResponse):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_conversion, body.template)
    except Exception as e:
        logger.error("Template conversion error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
