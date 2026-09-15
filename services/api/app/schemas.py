from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class WarningItem(BaseModel):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str


class InputSummary(BaseModel):
    sha256: str
    format: Literal["PNG", "JPEG"]
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    color_mode: Literal["grayscale"] = "grayscale"


class OaPrediction(BaseModel):
    model_version: str
    selected_kl_grade: int = Field(ge=0, le=4)
    probabilities: dict[str, float]
    uncertainty: float = Field(ge=0, le=1)


class AnalysisResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    study_id: str
    status: Literal["completed"] = "completed"
    research_only: Literal[True] = True
    created_at: datetime
    input: InputSummary
    pipeline_version: str
    prediction: OaPrediction
    warnings: list[WarningItem]

