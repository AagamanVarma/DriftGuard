"""Simple FastAPI server for text classification."""

from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

from app.core import ModelStore, build_drift_baseline, detect_text_drift, setup_logger, train_and_promote


logger = setup_logger(__name__)


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


class IngestResponse(BaseModel):
    received: int
    drift: Dict[str, Any]
    retrained: bool
    new_version: Optional[str] = None
    message: str


app = FastAPI(
    title="Student ML Project API",
    description="Simple text classification API with model versioning",
    version="1.0.0",
)

store: Optional[ModelStore] = None
current_model: Optional[Any] = None
current_vectorizer: Optional[Any] = None
current_version: Optional[str] = None
project_root: Optional[Path] = None
dataset_path: Optional[Path] = None
model_lock = Lock()


def load_version(version: str) -> None:
    """Load one model version into memory and mark it production."""
    global current_model, current_vectorizer, current_version

    if store is None:
        raise RuntimeError("Model store not initialized")

    if not store.exists(version):
        raise FileNotFoundError(f"Model {version} not found")

    model = store.load_model(version)
    vectorizer = store.load_vectorizer(version)

    with model_lock:
        store.set_production(version)
        current_model = model
        current_vectorizer = vectorizer
        current_version = version


@app.on_event("startup")
async def startup() -> None:
    """Initialize model store and try loading production model."""
    global store, project_root, dataset_path

    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "datasets" / "sample_data.csv"
    store = ModelStore(project_root / "models", project_root / "production")

    try:
        version = store.get_production()
        load_version(version)
        logger.info(f"Loaded production model: {version}")
    except FileNotFoundError:
        logger.warning("No production model set yet. Train first with scripts/train.py")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if current_model is not None else "unhealthy",
        model_loaded=current_model is not None,
        current_model=current_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    if current_model is None or current_vectorizer is None or current_version is None:
        raise HTTPException(status_code=503, detail="No production model loaded")

    try:
        X = current_vectorizer.transform([request.text]).toarray()

        with model_lock:
            pred = current_model.predict(X)[0]

            probs = None
            pred_prob = None
            if request.return_probabilities:
                if hasattr(current_model, "predict_proba"):
                    raw_probs = current_model.predict_proba(X)[0]
                    classes = current_model.classes_
                    probs = {str(c): float(p) for c, p in zip(classes, raw_probs)}
                    pred_prob = probs.get(str(pred))
                else:
                    probs = {str(pred): 1.0}
                    pred_prob = 1.0

        return PredictionResponse(
            text=request.text,
            prediction=str(pred),
            probability=pred_prob,
            model_version=current_version,
            probabilities=probs,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    if store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    models = store.list_models()
    return {"total": len(models), "models": models}


@app.get("/models/current")
async def current_model_info() -> Dict[str, Any]:
    if store is None or current_version is None:
        raise HTTPException(status_code=404, detail="No production model set")

    return {
        "version": current_version,
        "metrics": store.load_metrics(current_version),
        "config": store.load_config(current_version),
    }


@app.post("/models/{version}/promote")
async def promote_model(version: str) -> Dict[str, str]:
    try:
        load_version(version)
        return {"message": f"Model {version} promoted to production"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {version} not found")
    except Exception as e:
        logger.error(f"Promotion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{version}/rollback")
async def rollback_model(version: str) -> Dict[str, str]:
    """In this simple project rollback is equivalent to promote old version."""
    return await promote_model(version)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(request: IngestRequest) -> IngestResponse:
    """Ingest incoming labeled data, detect drift, and optionally auto-retrain."""
    if store is None or current_version is None:
        raise HTTPException(status_code=503, detail="No production model loaded")
    if dataset_path is None:
        raise HTTPException(status_code=503, detail="Dataset path not initialized")
    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided")

    texts = [r.text for r in request.records]
    labels = [r.label for r in request.records if r.label is not None]

    config = store.load_config(current_version)
    baseline = config.get("drift_baseline")
    if baseline is None:
        # Backward compatibility with older saved models
        try:
            ref_df = pd.read_csv(dataset_path)
            baseline = build_drift_baseline(
                texts=ref_df["text"].astype(str).tolist(),
                labels=ref_df["label"].astype(str).tolist(),
                vectorizer=current_vectorizer,
            )
        except Exception as e:
            logger.warning(f"Could not rebuild drift baseline from dataset: {e}")
            baseline = {
                "length_mean": 1.0,
                "label_distribution": {},
                "vocabulary": [],
            }

    drift = detect_text_drift(
        incoming_texts=texts,
        baseline=baseline,
        incoming_labels=labels if len(labels) == len(texts) else None,
        threshold=request.drift_threshold,
    )

    if not drift["is_drift"]:
        return IngestResponse(
            received=len(request.records),
            drift=drift,
            retrained=False,
            message="No meaningful drift detected. Production model unchanged.",
        )

    if not request.auto_retrain:
        return IngestResponse(
            received=len(request.records),
            drift=drift,
            retrained=False,
            message="Drift detected, but auto_retrain is disabled.",
        )

    if len(labels) != len(texts):
        return IngestResponse(
            received=len(request.records),
            drift=drift,
            retrained=False,
            message="Drift detected but retraining skipped: every record needs a label.",
        )

    try:
        df_existing = pd.read_csv(dataset_path)
        next_id = int(df_existing["id"].max()) + 1 if not df_existing.empty else 1

        new_rows = []
        for rec in request.records:
            rid = rec.id if rec.id is not None else next_id
            next_id += 1 if rec.id is None else 0
            new_rows.append({"id": rid, "text": rec.text, "label": rec.label})

        df_new = pd.DataFrame(new_rows)
        merged = pd.concat([df_existing, df_new], ignore_index=True)
        merged.to_csv(dataset_path, index=False)

        result = train_and_promote(
            dataset_path=dataset_path,
            models_dir=project_root / "models",
            production_dir=project_root / "production",
            optimize=True,
            n_trials=5,
            random_state=42,
        )

        load_version(result["version"])

        return IngestResponse(
            received=len(request.records),
            drift=drift,
            retrained=True,
            new_version=result["version"],
            message=(
                "Drift detected. Incoming data appended, model retrained, and "
                f"{result['version']} promoted to production."
            ),
        )
    except Exception as e:
        logger.error(f"Ingest/retrain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest/retrain failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
