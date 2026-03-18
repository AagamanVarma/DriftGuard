# Drift aware ML system

This is an ML project that:

1. trains multiple text-classification models,
2. picks the best one,
3. saves it with versioning, and
4. serves predictions through a FastAPI API,
5. detects incoming data drift, and
6. auto-retrains + upgrades when drift is high.



## Project Structure

```
project/
├── app/
│   ├── core.py              # Main logic (data, training, metrics, model store)
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py        # FastAPI server
│   └── __init__.py
├── scripts/
│   └── train.py             # Train + select best + save + promote
├── datasets/
│   └── sample_data.csv      # Input CSV (id, text, label)
├── models/                  # Saved model versions (model_v1, model_v2...)
├── production/
│   └── current_model.txt    # Current production version pointer
├── requirements.txt
└── README.md
```

## Dataset Format

CSV must have exactly these columns:

- `id`
- `text`
- `label`

Example:

```csv
id,text,label
1,Great product quality,positive
2,Terrible service,negative
3,Amazing support,positive
```


## Train Model

```bash
python scripts/train.py
```

This will:

- load and clean dataset,
- split train/val/test,
- train LogisticRegression, RandomForest, LinearSVC,
- select the best by weighted score,
- save it to `models/model_vX/`,
- mark it as production in `production/current_model.txt`.

## Run API

```bash
python -m uvicorn app.api.server:app --reload --port 8000
```

## API Endpoints

- `GET /health` → health status
- `POST /predict` → predict label from text
- `GET /models` → list all saved models
- `GET /models/current` → show current production model info
- `POST /models/{version}/promote` → set specific version as production
- `POST /models/{version}/rollback` → same as promote (simple rollback)
- `POST /ingest` → ingest incoming batch, detect drift, and auto-retrain/promote if drift is detected

## Drift-aware Lifecycle

Main idea:

1. New incoming records arrive through `POST /ingest`.
2. System compares incoming text statistics with the current model's saved baseline:
   - text length shift,
   - out-of-vocabulary token rate,
   - label distribution shift (if labels are provided).
3. If drift score exceeds threshold (default `0.35`), drift is flagged.
4. If `auto_retrain=true` and all records include labels:
   - incoming data is appended to `datasets/sample_data.csv`,
   - training pipeline runs automatically,
   - best model is saved as new `model_vX`,
   - new version is promoted to production.

This gives: **Incoming data → drift detection → automatic upgrade**.

### Sample ingest request

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "auto_retrain": true,
    "drift_threshold": 0.35,
    "records": [
      {"text": "Packaging was awful and service was slow", "label": "negative"},
      {"text": "Brilliant quality and very fast delivery", "label": "positive"}
    ]
  }'
```

### Sample prediction request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing", "return_probabilities": true}'
```




