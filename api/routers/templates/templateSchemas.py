"""Pydantic schemas for template items.

Mirrors frontend/src/types/index.ts's template item types, which in turn
mirror the `anyOf` branches of claudeTemplates.ts's TEMPLATE_ITEM_SCHEMAS.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

ChordQuality = Literal[
    "maj7", "maj", "min", "dom7", "min7", "dim7", "dim", "m7b5", "sus2", "7sus", "7aug", "aug"
]
_QUALITY_ALIASES = {"7": "dom7", "m": "min", "m7": "min7", "": "maj", "o7": "dim7", "ø7": "m7b5", "sus": "7sus", "7+": "7aug", "+": "aug"}
_SCALE_DEGREES =dict(zip("1234567", ("I", "II", "III", "IV", "V", "VI", "VII")))


class TemplateItemBase(BaseModel):
    measure: int | None = Field(None, description="The 1-based measure this item belongs to. Omit if not yet placed.")
    pos: float | None = Field(None, description="The beat position within the measure this item belongs to. Omit if not yet placed.")


class TemplateMaskedBars(BaseModel):
    """Assigns a completely masked sequence of measures. This is the catalyst for completely new content.
    May also be used on a sub measure level by setting measure length to 1."""

    kind: Literal["masked"] = "masked"
    measure_start: int = Field(..., description="The beginning of a sequence of completely masked measures (1-based).")
    mask_measure_length: int = Field(..., description="The amount of measures to have completely masked.")
    slots_per_measure: int = Field(..., description="The maximum number of new elements filled in per masked measure. Not all slots will be used")


class InstrumentTemplateItem(TemplateItemBase):
    """Assigns/blanks an instrument for a voice."""

    kind: Literal["instrument"] = "instrument"
    instrument: str | None = Field(None, description="The instrument to assign. Omit to leave this item blank.")
    voice_idx: int = Field(..., description="The index of the voice this item belongs to.")


class ClefTemplateItem(TemplateItemBase):
    """Assigns/blanks a clef for a voice."""

    kind: Literal["clef"] = "clef"
    name: str | None = Field(None, description="The clef to assign. Omit to leave this item blank.")
    voice_idx: int = Field(..., description="The index of the voice this item belongs to.")


class KeyTemplateItem(TemplateItemBase):
    """Assigns/blanks a key signature."""

    kind: Literal["key"] = "key"
    key_tonic: str | None = Field(None, description="The tonic note of the key. Ex: C, Bb, E, F#. Omit to leave this item blank.")
    key_mode: str | None = Field(None, description="The mode of the key. Ex: major, minor, mixolydian. Omit to leave this item blank.")


class TimeTemplateItem(TemplateItemBase):
    """Assigns/blanks a time signature."""

    kind: Literal["time"] = "time"
    numerator: str | None = Field(None, description="The time signature's numerator. Omit to leave this item blank.")
    denominator: str | None = Field(None, description="The time signature's denominator. Omit to leave this item blank.")


class TempoTemplateItem(TemplateItemBase):
    """Assigns/blanks a tempo."""

    kind: Literal["tempo"] = "tempo"
    tempo: int | None = Field(None, description="The tempo in beats per minute. Omit to leave this item blank.")


class NoteTemplateItem(TemplateItemBase):
    """Assigns/blanks a single note."""

    kind: Literal["note"] = "note"
    duration: float | None = Field(None, description="The note's duration. Omit to leave this item blank.")
    pitch: int | None = Field(None, description="The note's pitch. Omit to leave this item blank.")
    voice_idx: int | None = Field(None, description="The index of the voice this note belongs to. Omit to leave this item blank.")


class ChordTemplateItem(TemplateItemBase):
    """Assigns/blanks a chord."""

    kind: Literal["chord"] = "chord"
    root: str | None = Field(None, description="The chord's root: either an absolute note name (G, Bb, F#) or a roman numeral relative to the key in effect (ii, V, bVII). Omit to leave this item blank.")
    quality: ChordQuality | None = Field(None, description="The chord's quality. Omit to leave this item blank.")

    @model_validator(mode="before")
    @classmethod
    def _legacy_root(cls, data):
        # a template written before `root` named its root as `degree` (a roman numeral, which kept
        # alterations) and/or `rootNumeral` (the plain scale degree, as a digit)
        if isinstance(data, dict) and data.get("root") is None:
            legacy = data.get("degree") or data.get("rootNumeral")
            if legacy is not None:
                legacy = str(legacy)
                if legacy in _SCALE_DEGREES: legacy = _SCALE_DEGREES[legacy]
                data = {**data, "root": legacy}
        # chord-symbol spellings of a quality ("7" for a dominant seventh) are accepted too
        if isinstance(data, dict) and data.get("quality") in _QUALITY_ALIASES:
            data = {**data, "quality": _QUALITY_ALIASES[data["quality"]]}
        return data


AnyTemplateItem = Annotated[
    Union[
        TemplateMaskedBars,
        InstrumentTemplateItem,
        ClefTemplateItem,
        KeyTemplateItem,
        TimeTemplateItem,
        TempoTemplateItem,
        NoteTemplateItem,
        ChordTemplateItem,
    ],
    Field(discriminator="kind"),
]


class TemplateResponse(BaseModel):
    template: list[AnyTemplateItem] = Field(..., description="Generated template content")


class FillTemplateRequest(BaseModel):
    template: list[AnyTemplateItem] = Field(..., description="Template items to fill in")
    steps: int = Field(12, ge=1, le=64, description="Decoding passes; more passes commit fewer slots at a time")
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    model_name: str = Field("the-waffiller", description="Model checkpoint name")
    device: str = Field("cpu", description="Inference device: cpu or cuda")