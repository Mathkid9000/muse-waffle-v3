"""POST /api/templates/fill-template — write a template's blanks and return it as a score."""

from __future__ import annotations
import asyncio
import logging
import traceback
from api.schemas import XmlResponse
from fastapi import APIRouter, HTTPException
from api.routers.edit.instr import _INSTR_DISPLAY
from api.routers.templates.score_to_template import _QUALITIES, _TEMPLATE_INSTRUMENTS
from api.routers.templates.templateSchemas import FillTemplateRequest

logger = logging.getLogger(__name__)
router = APIRouter()

_DISPLAY = dict(zip(_TEMPLATE_INSTRUMENTS, _INSTR_DISPLAY))
_FIGURES = {template: figure for figure, template in reversed(_QUALITIES.items())}
# the corpus never named a diminished seventh, so QUAL_o7 has no embedding and would be handed to the
# model as a MASK; the diminished triad is the nearest chord the model can actually be told
_CANVAS_FIGURES = {**_FIGURES, "dim7": "dim"}
# the order rows of the same beat are written in, mirroring tokenize_score_as_text's own priority;
# a masked block goes last, after whatever its first bar places before it
_KINDS = ("key", "time", "tempo", "clef", "chord", "note", "masked")


def _canvas(template) -> list[tuple[str, str, str, str, str]]:
    """A template as the rows the model reads: a header, the items in score order, and a MASK
    wherever the template left a field blank for the model to write."""
    import music21 as m21
    from infill import masked_bars
    from ml.api import _advance
    from tokenizer import SUBDIVISIONS_PER_QUARTER as SUB, canonical_instrument

    def tok(prefix: str, value) -> str:
        return "MASK" if value is None else f"{prefix}_{value}"

    def placement(item):
        # score order; an unplaced item sorts to the end, to be handed over for the model to position
        if item.kind == "masked":
            return item.measure_start, 0.0, _KINDS.index(item.kind)
        return item.measure or float("inf"), item.pos or 0.0, _KINDS.index(item.kind)

    rows: list[tuple[str, str, str, str, str]] = [("SOG", "", "", "", "")]
    for it in (it for it in template if it.kind == "instrument"):
        # named the way the tokenizer names a part, so the header lands on the token the corpus uses
        name = canonical_instrument(m21.instrument.Instrument(instrumentName=_DISPLAY.get(it.instrument, "Piano")))
        rows.append((tok("INSTR", name.lower() if it.instrument else None), "", "", "", f"VOICE_{it.voice_idx}"))
    rows.append(("SOS", "", "", "", ""))

    bar, key = 0, m21.key.Key("C")
    for it in sorted((it for it in template if it.kind != "instrument"), key=placement):
        if it.kind == "masked":
            block = masked_bars(it.mask_measure_length, it.slots_per_measure, start_next=False)
            block[0] = (_advance(it.measure_start - bar), *block[0][1:])
            rows.extend(block)
            bar = it.measure_start + it.mask_measure_length - 1
            continue

        advance = "MASK" if it.measure is None else _advance(it.measure - bar)
        bar = it.measure if it.measure is not None else bar
        pos = tok("POS", None if it.pos is None else round(it.pos * SUB))
        match it.kind:
            case "key":
                if it.key_tonic is not None:
                    key = m21.key.Key(it.key_tonic, it.key_mode or "major")
                # the key token is the signature's own tonic, ie. a minor key under its relative
                # major, which is how the tokenizer reads one back off a score
                rows.append((advance, pos, tok("KEY", key.asKey().tonic.pitchClass if it.key_tonic else None), "", ""))
            case "time":
                rows.append((advance, pos, tok("NUM", it.numerator), tok("DEN", it.denominator), ""))
            case "tempo":
                rows.append((advance, pos, tok("TEMPO", None if it.tempo is None else round(it.tempo / 16)), "", ""))
            case "clef":
                rows.append((advance, pos, tok("CLEF", it.name), "", f"VOICE_{it.voice_idx}"))
            case "note":
                rows.append((advance, pos, tok("DUR", None if it.duration is None else round(it.duration * SUB)),
                             tok("PITCH", it.pitch), tok("VOICE", it.voice_idx)))
            case "chord":
                # degree keeps alterations (bVII); rootNumeral is the plain scale degree, often given as a digit
                numeral = it.degree or it.rootNumeral
                if numeral and numeral.isdigit(): numeral = int(numeral)
                root = m21.roman.RomanNumeral(numeral, key).root().pitchClass if numeral else None
                degree = None if root is None else (root - key.asKey().tonic.pitchClass) % 12
                rows.append((advance, pos, tok("DEG", degree), tok("ROOT", root),
                             tok("QUAL", _CANVAS_FIGURES.get(it.quality) if it.quality else None)))

    rows.append(("EOS", "", "", "", ""))
    return rows


def _run_fill(body: FillTemplateRequest) -> str:
    from ml.api import fill_template

    if not any(it.kind == "instrument" for it in body.template):
        raise RuntimeError("Template fill failed: the template names no instrument to write for")

    return fill_template(
        _canvas(body.template),
        steps=body.steps,
        temperature=body.temperature,
        device=body.device,
        model_name=body.model_name,
        log=logger.info,
    )


@router.post("/templates/fill-template", response_model=XmlResponse)
async def fill(body: FillTemplateRequest):
    loop = asyncio.get_running_loop()
    try:
        xml = await loop.run_in_executor(None, _run_fill, body)
    except Exception as e:
        logger.error("Fill error:\n%s", traceback.format_exc())
        raise HTTPException(500, detail=str(e))
    return XmlResponse(xml=xml)
