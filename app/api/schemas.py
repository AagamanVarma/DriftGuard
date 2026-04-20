"""Pydantic schemas used by the API layer."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    return_probabilities: bool = Field(False, description="Return class probabilities")


class PredictionResponse(BaseModel):
    text: str
    prediction: str
    probability: Optional[float] = None
    model_version: str
    probabilities: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    current_model: Optional[str] = None


class IncomingRecord(BaseModel):
    id: Optional[int] = Field(None, description="Optional row identifier")
    text: str = Field(..., description="Incoming text sample")
    label: Optional[str] = Field(None, description="Optional ground-truth label")


class IngestRequest(BaseModel):
    records: list[IncomingRecord] = Field(..., description="Incoming batch records")
    auto_retrain: bool = Field(True, description="Retrain and promote if drift is detected")
    drift_threshold: float = Field(0.35, ge=0.0, le=1.0)
    min_window_size: int = Field(
        50,
        ge=1,
        description="Minimum batch size required before running drift detection/retraining",
    )


class IngestResponse(BaseModel):
    received: int
    drift: Dict[str, Any]
    retrained: bool
    new_version: Optional[str] = None
    champion_version: Optional[str] = None
    challenger_version: Optional[str] = None
    promoted: Optional[bool] = None
    comparison: Optional[Dict[str, Any]] = None
    message: str
