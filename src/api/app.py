from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.api.schemas import (
    Base64BatchRequest,
    Base64PredictRequest,
    PredictBatchResponse,
    PredictResponse,
    PredictionResult,
)
from src.inference.predictor import ImagePredictor
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger

config = load_config()
logger = setup_logger("api", Path(config["paths"]["logs_dir"]) / "api.log")
predictor: Optional[ImagePredictor] = None

app = FastAPI(title="Refund Image Classifier API", version="1.0.0")


def _get_predictor() -> ImagePredictor:
    global predictor
    if predictor is None:
        predictor = ImagePredictor(
            model_path=config["paths"]["model_file"],
            class_map_path=config["paths"]["class_map_file"],
            image_size=tuple(config["dataset"]["image_size"]),
        )
    return predictor


@app.get("/health")
def health() -> dict:
    model_path = Path(config["paths"]["model_file"])
    class_map_path = Path(config["paths"]["class_map_file"])
    artifacts_ready = model_path.exists() and class_map_path.exists()
    return {"status": "ok", "model_loaded": artifacts_ready}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: Base64PredictRequest) -> PredictResponse:
    try:
        pred = _get_predictor().predict_base64(payload.image_base64)
        result = PredictionResult(filename=payload.filename, **pred)
        return PredictResponse(result=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {exc}") from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc


@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(payload: Base64BatchRequest) -> PredictBatchResponse:
    results: List[PredictionResult] = []
    try:
        service = _get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {exc}") from exc

    for item in payload.images:
        try:
            pred = service.predict_base64(item.image_base64)
            results.append(PredictionResult(filename=item.filename, **pred))
        except Exception as exc:
            logger.warning("Skipping invalid batch item %s: %s", item.filename, exc)
    if not results:
        raise HTTPException(status_code=400, detail="No valid images were provided")
    return PredictBatchResponse(results=results)


@app.post("/predict-upload", response_model=PredictBatchResponse)
async def predict_upload(files: List[UploadFile] = File(...)) -> PredictBatchResponse:
    try:
        service = _get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {exc}") from exc

    results: List[PredictionResult] = []
    for file in files:
        try:
            content = await file.read()
            pred = service.predict_bytes(content)
            results.append(PredictionResult(filename=file.filename or "uploaded_file", **pred))
        except Exception as exc:
            logger.warning("Invalid uploaded image %s: %s", file.filename, exc)
    if not results:
        raise HTTPException(status_code=400, detail="No valid image uploads were provided")
    return PredictBatchResponse(results=results)


@app.post("/predict-upload-single", response_model=PredictResponse)
async def predict_upload_single(file: UploadFile = File(...)) -> PredictResponse:
    """Single-file upload endpoint kept for Swagger UI compatibility."""
    try:
        service = _get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {exc}") from exc

    try:
        content = await file.read()
        pred = service.predict_bytes(content)
        result = PredictionResult(filename=file.filename or "uploaded_file", **pred)
        return PredictResponse(result=result)
    except Exception as exc:
        logger.exception("Invalid uploaded image %s", file.filename)
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc
