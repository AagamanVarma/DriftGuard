#!/usr/bin/env python3
"""
Simple training script (student-project style).
Flow:
1. Load CSV
2. Split + vectorize text
3. Train candidate models
4. Pick best one
5. Save as new version and mark production
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core import (
    setup_logger,
    train_and_promote,
)


logger = setup_logger(__name__)


def main():
    """Main training pipeline"""
    logger.info("=" * 80)
    logger.info("SELF-IMPROVING ML SYSTEM - INITIAL TRAINING")
    logger.info("=" * 80)
    
    # Project folders used by this script.
    dataset_path = project_root / "datasets" / "sample_data.csv"
    models_dir = project_root / "models"
    production_dir = project_root / "production"
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    try:
        logger.info("\n[Step 1-4] Training + selecting + saving + promoting best model...")
        # optimize=True here means: try a small set of readable configs,
        # then pick the best one using validation metrics.
        result = train_and_promote(
            dataset_path=dataset_path,
            models_dir=models_dir,
            production_dir=production_dir,
            optimize=True,
            n_trials=10,
            random_state=42,
        )

        logger.info(f"Class distribution: {result['class_distribution']}")
        logger.info(f"Best model: {result['best_model_type']} (score: {result['best_score']:.4f})")
        logger.info(f"Model {result['version']} is now in production")
        logger.info(
            "Metrics: "
            f"acc={result['metrics']['accuracy']:.4f}, "
            f"f1={result['metrics']['f1_score']:.4f}, "
            f"precision={result['metrics']['precision']:.4f}, "
            f"recall={result['metrics']['recall']:.4f}, "
            f"latency_ms={result['metrics']['latency_ms']:.4f}"
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("INITIAL TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("1. Start the API server: python -m app.api.server")
        logger.info("2. Send prediction requests to /predict")
        logger.info("3. Send new batch data to /ingest for drift-aware auto-upgrades")
        
        return True
    
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
