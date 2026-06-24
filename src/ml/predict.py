"""
Credit default inference: probability, risk band, and approve/reject decision.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd

from src.data.loader import load_metrics, load_model
from src.data.preprocessor import PreprocessingError, preprocess_applicant
from src.utils.config import RISK_BAND_LOW_MAX, RISK_BAND_MEDIUM_MAX

logger = logging.getLogger(__name__)

RiskBand = Literal["LOW", "MEDIUM", "HIGH"]
Decision = Literal["APPROVE", "REJECT"]


def get_threshold() -> float:
    """Return the optimal classification threshold from saved metrics."""
    metrics = load_metrics()
    return float(metrics["threshold"])


def assign_risk_band(probability: float, threshold: float) -> RiskBand:
    """
    Map default probability to a risk band.

    LOW    : probability < 0.40
    MEDIUM : 0.40 <= probability < threshold
    HIGH   : probability >= threshold (aligned with reject zone)
    """
    medium_cutoff = min(RISK_BAND_MEDIUM_MAX, threshold)
    low_cutoff = min(RISK_BAND_LOW_MAX, medium_cutoff)

    if probability < low_cutoff:
        return "LOW"
    if probability < medium_cutoff:
        return "MEDIUM"
    return "HIGH"


def assign_decision(probability: float, threshold: float) -> Decision:
    """APPROVE if default probability is below the tuned threshold; otherwise REJECT."""
    return "REJECT" if probability >= threshold else "APPROVE"


def predict_default_probability(model_input: pd.DataFrame) -> float:
    """Run the loaded model and return P(default=1)."""
    model = load_model()
    try:
        probabilities = model.predict_proba(model_input)
        return float(probabilities[0][1])
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise RuntimeError(f"Prediction failed: {exc}") from exc


def predict_applicant(data: dict[str, Any]) -> dict[str, Any]:
    """
    Full inference pipeline for one applicant.

    Returns default probability, risk band, decision, and the threshold used.
    """
    threshold = get_threshold()
    model_input = preprocess_applicant(data)
    probability = predict_default_probability(model_input)
    risk_band = assign_risk_band(probability, threshold)
    decision = assign_decision(probability, threshold)

    logger.info(
        "Prediction complete | probability=%.4f risk=%s decision=%s",
        probability,
        risk_band,
        decision,
    )

    return {
        "default_probability": round(probability, 4),
        "risk_band": risk_band,
        "decision": decision,
        "threshold": threshold,
    }


def predict_risk(data: dict[str, Any]) -> dict[str, Any]:
    """
    Backward-compatible alias used by early API stubs.

    Maps the new response shape to legacy keys: probability, risk.
    """
    result = predict_applicant(data)
    return {
        "probability": result["default_probability"],
        "risk": result["risk_band"],
        "decision": result["decision"],
        "threshold": result["threshold"],
    }


__all__ = [
    "predict_applicant",
    "predict_risk",
    "predict_default_probability",
    "assign_risk_band",
    "assign_decision",
    "get_threshold",
]
