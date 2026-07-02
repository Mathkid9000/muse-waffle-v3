"""Pydantic request/response schemas for the Muse Waffle API."""

import base64
import io
import xml.etree.ElementTree as _ET
import zipfile

from pydantic import BaseModel, Field, field_validator


def _decode_xml(v: str) -> str:
    """Accept raw MusicXML or the frontend's __MXL__<base64> ZIP format."""
    if not v.startswith("__MXL__"):
        return v
    data = base64.b64decode(v[7:])
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Prefer the rootfile declared in META-INF/container.xml
        if "META-INF/container.xml" in zf.namelist():
            container = _ET.fromstring(zf.read("META-INF/container.xml"))
            ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
            rootfile = container.find(".//c:rootfile", ns)
            if rootfile is not None:
                path = rootfile.get("full-path", "")
                if path and path in zf.namelist():
                    return zf.read(path).decode("utf-8")
        # Fallback: first .xml entry that isn't in META-INF
        for name in zf.namelist():
            if name.endswith(".xml") and not name.startswith("META-INF"):
                return zf.read(name).decode("utf-8")
    raise ValueError("Could not find MusicXML content in MXL archive")

# Creative Requests
class InfillRequest(BaseModel):
    xml: str = Field(..., description="MusicXML content as a string")

    @field_validator("xml", mode="before")
    @classmethod
    def resolve_xml(cls, v: str) -> str:
        return _decode_xml(v)
    start_measure: int = Field(..., ge=1, description="First measure of the gap (inclusive)")
    end_measure: int = Field(..., ge=1, description="Last measure of the gap (inclusive)")
    parts: list[int] | None = Field(
        None,
        description="0-indexed part indices to fill; None fills all parts",
    )
    temperature: float = Field(0.9, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=200)
    model_name: str = Field("INFILL-D1024-H16-L8", description="Infill model checkpoint name")
    device: str = Field("cpu", description="Inference device: cpu or cuda")


class GenerateRequest(BaseModel):
    instruments: list[int] = Field(
        [1],
        description="List of instrument indices (0-31, see INSTR_NAMES in ml/tokenizer.py)",
    )
    key: int = Field(6, ge=0, le=12, description="Key signature index (0=6 flats … 6=C … 12=6 sharps)")
    time_sig: int = Field(2, ge=0, le=5, description="Time signature index (0=2/4, 1=3/4, 2=4/4, 3=6/8, 4=3/8, 5=12/8)")
    tempo: int = Field(5, ge=0, le=15, description="Tempo bucket index (0=slowest … 15=fastest)")
    num_bars: int = Field(16, ge=1, le=128)
    max_new: int = Field(2000, ge=100, le=4096)
    temperature: float = Field(0.9, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=200)
    model_name: str = Field("GEN-D1024-H16-L8", description="Generation model checkpoint name")
    device: str = Field("cpu", description="Inference device: cpu or cuda")


class AppendRequest(BaseModel):
    xml: str = Field(..., description="Existing MusicXML content to append to")

    @field_validator("xml", mode="before")
    @classmethod
    def resolve_xml(cls, v: str) -> str:
        return _decode_xml(v)
    num_bars: int = Field(4, ge=1, le=32)
    max_new: int = Field(1024, ge=100, le=4096)
    temperature: float = Field(0.9, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=200)
    model_name: str = Field("GEN-D1024-H16-L8", description="Generation model checkpoint name")
    device: str = Field("cpu", description="Inference device: cpu or cuda")


# Deterministic requests
class EditClefRequest(BaseModel):
    xml: str = Field(..., description="Existing MusicXML content to change clef of")

    @field_validator("xml", mode="before")
    @classmethod
    def resolve_xml(cls, v: str) -> str:
        return _decode_xml(v)
    measure_num: int = Field(1, ge=1)
    staff_num: int = Field(0, ge=0)
    new_clef: int = Field(0, ge=0, lt=7)


class EditTimeSigRequest(BaseModel):
    xml: str = Field(..., description="Existing MusicXML content to change the time signature of")

    @field_validator("xml", mode="before")
    @classmethod
    def resolve_xml(cls, v: str) -> str:
        return _decode_xml(v)
    measure_num: int = Field(1, ge=1)
    new_time: int = Field(2, ge=0, lt=6, description="Time signature index (0=2/4, 1=3/4, 2=4/4, 3=6/8, 4=3/8, 5=12/8)")


class EditKeySigRequest(BaseModel):
    xml: str = Field(..., description="Existing MusicXML content to change the key signature of")

    @field_validator("xml", mode="before")
    @classmethod
    def resolve_xml(cls, v: str) -> str:
        return _decode_xml(v)
    measure_num: int = Field(1, ge=1)
    new_key: int = Field(6, ge=0, le=12, description="Key signature index (0=6 flats … 6=C … 12=6 sharps)")


class EditTempoRequest(BaseModel):
    xml: str = Field(..., description="Existing MusicXML content to change the tempo of")

    @field_validator("xml", mode="before")
    @classmethod
    def resolve_xml(cls, v: str) -> str:
        return _decode_xml(v)
    measure_num: int = Field(1, ge=1)
    new_tempo: int = Field(5, ge=0, lt=16, description="Tempo bucket index (0=slowest … 15=fastest)")


class EditInstrRequest(BaseModel):
    xml: str = Field(..., description="Existing MusicXML content to add an instrument to")

    @field_validator("xml", mode="before")
    @classmethod
    def resolve_xml(cls, v: str) -> str:
        return _decode_xml(v)
    instr: int = Field(1, ge=0, lt=32, description="Instrument index (0-31, see INSTR_NAMES in ml/tokenizer.py)")
    new_clef: int = Field(0, ge=0, lt=7, description="Clef for the new staff (0=treble … 6=bass8vb)")


class EditRmMeasureRequest(BaseModel):
    xml: str = Field(..., description="Existing MusicXML content to remove measures from")

    @field_validator("xml", mode="before")
    @classmethod
    def resolve_xml(cls, v: str) -> str:
        return _decode_xml(v)
    start_measure: int = Field(..., ge=1, description="First measure to remove (inclusive)")
    end_measure: int = Field(..., ge=1, description="Last measure to remove (inclusive)")


class XmlResponse(BaseModel):
    xml: str = Field(..., description="Generated MusicXML content")
