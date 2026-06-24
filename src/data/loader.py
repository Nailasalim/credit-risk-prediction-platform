"""
Load trained model artifacts for inference.

Artifacts are read once and cached in memory for fast repeated predictions.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import joblib
from lightgbm import LGBMClassifier

from src.utils.config import FEATURE_NAMES_PATH, METRICS_PATH, MODEL_PATH

logger = logging.getLogger(__name__)


class ArtifactLoadError(Exception):
    """Raised when a required model artifact cannot be loaded."""


@lru_cache(maxsize=1)
def load_model() -> LGBMClassifier:
    """Load the serialized LightGBM classifier from disk."""
    try:
        logger.info("Loading model from %s", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict_proba"):
            raise ArtifactLoadError(
                f"Object at {MODEL_PATH} is not a scikit-learn compatible classifier."
            )
        return model
    except FileNotFoundError as exc:
        logger.error("Model file not found: %s", MODEL_PATH)
        raise ArtifactLoadError(f"Model file not found: {MODEL_PATH}") from exc
    except Exception as exc:
        logger.exception("Failed to load model from %s", MODEL_PATH)
        raise ArtifactLoadError(f"Failed to load model: {exc}") from exc


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    """Load evaluation metrics and the optimal classification threshold."""
    try:
        logger.info("Loading metrics from %s", METRICS_PATH)
        with METRICS_PATH.open(encoding="utf-8") as file:
            metrics: dict[str, Any] = json.load(file)
        if "threshold" not in metrics:
            raise ArtifactLoadError(
                f"metrics.json must contain a 'threshold' key: {METRICS_PATH}"
            )
        return metrics
    except FileNotFoundError as exc:
        logger.error("Metrics file not found: %s", METRICS_PATH)
        raise ArtifactLoadError(f"Metrics file not found: {METRICS_PATH}") from exc
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in metrics file: %s", METRICS_PATH)
        raise ArtifactLoadError(f"Invalid metrics JSON: {exc}") from exc
    except Exception as exc:
        logger.exception("Failed to load metrics from %s", METRICS_PATH)
        raise ArtifactLoadError(f"Failed to load metrics: {exc}") from exc


@lru_cache(maxsize=1)
def load_feature_names() -> list[str]:
    """Load the ordered list of feature names expected by the model."""
    try:
        logger.info("Loading feature names from %s", FEATURE_NAMES_PATH)
        with FEATURE_NAMES_PATH.open(encoding="utf-8") as file:
            feature_names = json.load(file)
        if not isinstance(feature_names, list) or not feature_names:
            raise ArtifactLoadError(
                f"feature_names.json must be a non-empty list: {FEATURE_NAMES_PATH}"
            )
        return [str(name) for name in feature_names]
    except FileNotFoundError as exc:
        logger.error("Feature names file not found: %s", FEATURE_NAMES_PATH)
        raise ArtifactLoadError(
            f"Feature names file not found: {FEATURE_NAMES_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in feature names file: %s", FEATURE_NAMES_PATH)
        raise ArtifactLoadError(f"Invalid feature_names JSON: {exc}") from exc
    except Exception as exc:
        logger.exception("Failed to load feature names from %s", FEATURE_NAMES_PATH)
        raise ArtifactLoadError(f"Failed to load feature names: {exc}") from exc
