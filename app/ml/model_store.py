"""Model persistence and production pointer helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class ModelStore:
    """Very simple file-based model store with production pointer."""

    def __init__(self, models_dir: Path, production_dir: Path):
        self.models_dir = Path(models_dir)
        self.production_dir = Path(production_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.current_model_file = self.production_dir / "current_model.txt"

    def get_next_version(self) -> str:
        versions: List[int] = []
        for d in self.models_dir.iterdir():
            if d.is_dir() and d.name.startswith("model_v"):
                try:
                    versions.append(int(d.name.replace("model_v", "")))
                except ValueError:
                    continue
        return f"model_v{(max(versions) + 1) if versions else 1}"

    def save(
        self,
        model: Any,
        vectorizer: TfidfVectorizer,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        version: str | None = None,
    ) -> str:
        version = version or self.get_next_version()
        model_dir = self.models_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_dir / "model.pkl")
        joblib.dump(vectorizer, model_dir / "vectorizer.pkl")

        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return version

    def exists(self, version: str) -> bool:
        return (self.models_dir / version / "model.pkl").exists()

    def load_model(self, version: str) -> Any:
        return joblib.load(self.models_dir / version / "model.pkl")

    def load_vectorizer(self, version: str) -> TfidfVectorizer:
        return joblib.load(self.models_dir / version / "vectorizer.pkl")

    def load_metrics(self, version: str) -> Dict[str, Any]:
        with open(self.models_dir / version / "metrics.json", "r") as f:
            return json.load(f)

    def load_config(self, version: str) -> Dict[str, Any]:
        with open(self.models_dir / version / "config.json", "r") as f:
            return json.load(f)

    def set_production(self, version: str) -> None:
        if not self.exists(version):
            raise FileNotFoundError(f"Model {version} not found")
        with open(self.current_model_file, "w") as f:
            f.write(version)

    def get_production(self) -> str:
        if not self.current_model_file.exists():
            raise FileNotFoundError("No production model set")
        with open(self.current_model_file, "r") as f:
            return f.read().strip()

    def list_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        try:
            production_version = self.get_production()
        except FileNotFoundError:
            production_version = None

        for d in sorted(self.models_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("model_v"):
                continue
            try:
                models.append(
                    {
                        "version": d.name,
                        "is_production": d.name == production_version,
                        "metrics": self.load_metrics(d.name),
                        "config": self.load_config(d.name),
                    }
                )
            except Exception:
                continue
        return models
