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

# a still-unpitched note sits on the middle line of its staff, as a slashed notehead
MIDDLE_LINE = {'treble': 71, 'bass': 50, 'alto': 60, 'tenor': 57, 'treble8vb': 59, 'bass8vb': 38}


def _blank(kind: str) -> expr21.TextExpression:
    """The placeholder for wherever a template item has left its value(s) unset. There is no
    'blank clef' or 'blank key signature' in notation, so a still-empty field is named in text
    instead -- the same idea fill_template.py's canvas rows use a MASK token for."""
    text = expr21.TextExpression(f'{kind}?')
    text.style.fontStyle = 'italic'
    return text


def _run_conversion(template: list[AnyTemplateItem]) -> str:
    import notation
    from notation import SUBDIVISIONS_PER_QUARTER as SUB

    if not any(it.kind == "instrument" for it in template):
        raise RuntimeError("Template conversion failed: the template names no instrument to write for")

    score = stream.Score()
    parts: dict[int, stream.Part] = {}
    for item in template:
        if item.kind == 'instrument':
            parts[item.voice_idx] = stream.Part()
            parts[item.voice_idx].append(instrument21.Instrument(instrumentName=item.instrument or None))
            if not item.instrument:
                parts[item.voice_idx].partName = 'instrument?'
    if not parts:
        raise RuntimeError("Template conversion failed: every instrument slot is still blank")
    top_voice = min(parts)   # tempo marks and chord symbols are written once, above the top staff

    def _pos(it) -> float:
        return 0.0 if it.kind == 'masked' else (it.pos or 0.0)

    def _at(it):
        return it.measure_start if it.kind == 'masked' else it.measure

    # notes become the same events the model's rows do, and go through the same engraving, so a
    # template previews exactly the way its filled score will render. A note with no pitch yet is
    # a slashed notehead on the middle line; several of those in one slot, or a blank duration,
    # say so in a lyric since nothing else distinguishes them
    events: dict[int, dict[int, list[notation.Event]]] = {}   # events[voice][measure]
    slots: dict[tuple[int, int, float], list] = {}
    clef_of = {it.voice_idx: it.name for it in template if it.kind == 'clef' and it.name}
    for item in template:
        if item.kind == 'note' and item.voice_idx in parts and item.measure is not None:
            slots.setdefault((item.voice_idx, item.measure, item.pos or 0.0), []).append(item)
    for uid, ((voice_idx, m_num, pos), items) in enumerate(slots.items()):
        pitches = [it.pitch for it in items if it.pitch is not None]
        dur = next((it.duration for it in items if it.duration is not None), None)
        labels = []
        if not pitches and len(items) > 1: labels.append(f'×{len(items)}')
        if dur is None: labels.append('dur?')
        notes = [notation.NoteIn(p) for p in pitches] or [notation.NoteIn(MIDDLE_LINE.get(clef_of.get(voice_idx), 71), notehead='slash')]
        notes[0].label = ' '.join(labels) or None
        events.setdefault(voice_idx, {}).setdefault(m_num, []).append(
            notation.Event(round(pos * SUB), round((dur if dur is not None else 1.0) * SUB), notes, uid=uid))

    # key, time, tempo and chords are score-wide rather than per-voice: key and time are written
    # into every staff, the way a real score repeats them on each one, while tempo marks and chord
    # symbols land once, above the top staff. Roman-numeral chords need the key in effect at that
    # point, so these are walked in score order, tracking the most recently placed key as we go.
    placed: dict[tuple[int, int], list] = {}   # (voice, measure) -> [(offset, element)]
    measure_keys: dict[int, int] = {}
    measure_meters: dict[int, tuple[int, int]] = {}
    chords = notation.ChordContext()
    score_wide = sorted(
        (it for it in template if it.kind in ('key', 'time', 'tempo', 'chord', 'masked') and _at(it) is not None),
        key=lambda it: (_at(it), _pos(it)),
    )
    current_key = key21.Key('C')
    for it in score_wide:
        if it.kind == 'key':
            if it.key_tonic and it.key_mode:
                current_key = key21.Key(it.key_tonic, it.key_mode)
                measure_keys[it.measure] = current_key.sharps
                for voice_idx in parts:
                    placed.setdefault((voice_idx, it.measure), []).append((_pos(it), key21.KeySignature(current_key.sharps)))
            else:
                for voice_idx in parts:
                    placed.setdefault((voice_idx, it.measure), []).append((_pos(it), _blank('key')))
        elif it.kind == 'time':
            if it.numerator and it.denominator:
                measure_meters[it.measure] = (int(it.numerator), int(it.denominator))
                for voice_idx in parts:
                    placed.setdefault((voice_idx, it.measure), []).append(
                        (_pos(it), meter21.TimeSignature(f'{it.numerator}/{it.denominator}')))
            else:
                for voice_idx in parts:
                    placed.setdefault((voice_idx, it.measure), []).append((_pos(it), _blank('time')))
        elif it.kind == 'tempo':
            el = tempo21.MetronomeMark(number=it.tempo) if it.tempo is not None else _blank('tempo')
            placed.setdefault((top_voice, it.measure), []).append((_pos(it), el))
        elif it.kind == 'chord':
            if it.degree and it.quality:
                root = roman21.RomanNumeral(it.degree, current_key).root().name
                el = harmony21.ChordSymbol(root + _FIGURES.get(it.quality, ''))
                chords.add(it.measure, round(_pos(it) * SUB), el)
            else:
                el = _blank('chord')
            placed.setdefault((top_voice, it.measure), []).append((_pos(it), el))
        elif it.kind == 'masked':
            # the whole span is unwritten -- only the model can choose what actually goes there --
            # so it is left as empty bars in every voice, marked where the span begins
            for voice_idx in parts:
                for offset in range(it.mask_measure_length):
                    placed.setdefault((voice_idx, it.measure_start + offset), [])
                placed[(voice_idx, it.measure_start)].append((0.0, _blank('masked')))

    # clefs: per-voice, either the named clef or a blank marker on that voice's own staff
    for item in template:
        if item.kind == 'clef' and item.voice_idx in parts and item.measure is not None:
            el = clef21.clefFromString(item.name) if item.name else _blank('clef')
            placed.setdefault((item.voice_idx, item.measure), []).append((item.pos or 0.0, el))

    all_measure_nums = sorted({m for v in events for m in events[v]} | {m for _, m in placed})
    if not all_measure_nums:
        raise RuntimeError("Template conversion failed: nothing in the template renders yet")
    first, last = all_measure_nums[0], all_measure_nums[-1]
    if not measure_meters:
        measure_meters[first] = (4, 4)
        for voice_idx in parts:
            placed.setdefault((voice_idx, first), []).insert(0, (0.0, meter21.TimeSignature('4/4')))
    meters: dict[int, tuple[int, int]] = {}
    sharps: dict[int, int | None] = {}
    current_meter, current_sharps = (4, 4), None
    for m in range(first, last + 2):
        current_meter = measure_meters.get(m, current_meter)
        current_sharps = measure_keys.get(m, current_sharps)
        meters[m], sharps[m] = current_meter, current_sharps

    # every part gets every bar, padded with measure rests where a voice has nothing written, so
    # the staves line up
    for voice_idx in sorted(parts):
        staff_voices = notation.layout_staff(events.get(voice_idx, {}), meters, first, last)
        for m in range(first, last + 1):
            measure = stream.Measure(number=m)
            for offset, el in placed.get((voice_idx, m), []):
                measure.insert(offset, el)
            num, den = meters[m]
            notation.fill_measure(measure, staff_voices.get(m, []), num, den, m, sharps[m], chords,
                                  label_text=lambda labels: ' '.join(str(l) for l in labels if l))
            parts[voice_idx].append(measure)
        score.insert(0, parts[voice_idx])

    return notation.finish(score)


@router.post("/templates/template-to-xml", response_model=XmlResponse)
async def convert(body: TemplateResponse):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_conversion, body.template)
    except Exception as e:
        logger.error("Template conversion error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
