"""Pydantic schemas for template items.

Mirrors frontend/src/types/index.ts's template item types, which in turn
mirror the `anyOf` branches of claudeTemplates.ts's TEMPLATE_ITEM_SCHEMAS.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

ChordQuality = Literal[
    "maj7", "maj", "min", "dom7", "min7", "dim7", "dim", "m7b5", "sus2", "7sus", "7aug", "aug"
]


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
    degree: str | None = Field(None, description="The chord's roman numeral degree. Omit to leave this item blank.")
    rootNumeral: str | None = Field(None, description="The numeral of the chord's root degree. Omit to leave this item blank.")
    quality: ChordQuality | None = Field(None, description="The chord's quality. Omit to leave this item blank.")


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
