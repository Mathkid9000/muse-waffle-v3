"""Pydantic request/response schemas for the Muse Waffle API."""

from pydantic import BaseModel, Field


class InfillRequest(BaseModel):
    xml: str = Field(..., description="MusicXML content as a string")
    start_measure: int = Field(..., ge=1, description="First measure of the gap (inclusive)")
    end_measure: int = Field(..., ge=1, description="Last measure of the gap (inclusive)")
    temperature: float = Field(0.9, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=200)
    model_name: str = Field("D512-H8-L8", description="Infill model checkpoint name")
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
    model_name: str = Field("GEN-D1024-H8-L8", description="Generation model checkpoint name")
    device: str = Field("cpu", description="Inference device: cpu or cuda")


class XmlResponse(BaseModel):
    xml: str = Field(..., description="Generated MusicXML content")
