# DriftGuard: Drift aware ML system

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

## Quick Start (3 steps)

1. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

2. **Train initial model:**
  ```bash
  python scripts/train.py
  ```

3. **Start API server (in new terminal):**
  ```bash
  docker-compose up --build
  ```

Server runs at `http://localhost:8000`

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
- try a small set of readable parameter candidates for each model,
- select the best by weighted score,
- save it to `models/model_vX/`,
- mark it as production in `production/current_model.txt`.

## Run API

### Option 1: Docker (build locally)

**Build and run with Docker Compose:**

```bash
docker-compose up --build
```

This will:
- Build the Docker image
- Start the FastAPI server on `http://localhost:8000`
- Mount local `datasets/`, `models/`, and `production/` directories for persistence

**Or build manually:**

```bash
docker build -t ml-api .
docker run -p 8000:8000 -v $(pwd)/datasets:/app/datasets -v $(pwd)/models:/app/models -v $(pwd)/production:/app/production ml-api
```

### Option 2: Run from Docker Hub image

Use the published image directly:

```bash
docker pull aagamanv/driftguard-api:latest
docker run -p 8000:8000 \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/production:/app/production \
  aagamanv/driftguard-api:latest
```

Health check:

```bash
curl http://localhost:8000/health
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




