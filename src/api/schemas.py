from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class Base64PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image payload")
    filename: str = Field(default="image")


class Base64BatchRequest(BaseModel):
    images: List[Base64PredictRequest]


class PredictionResult(BaseModel):
    filename: str
    predicted_label: str
    predicted_index: int
    probabilities: Dict[str, float]


class PredictResponse(BaseModel):
    result: PredictionResult


class PredictBatchResponse(BaseModel):
    results: List[PredictionResult]
