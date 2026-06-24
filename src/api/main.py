"""
FastAPI entry point for credit risk inference.

Run from project root:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.dashboard_summary import build_dashboard_summary
from src.data.preprocessor import PreprocessingError
from src.ml.predict import predict_applicant
from src.ml.rules import build_decision, list_active_rules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Inference API for the Home Credit default risk model.",
    version="1.0.0",
)


class ApplicantRequest(BaseModel):
    """Raw applicant fields (engineered ratios are computed server-side)."""

    model_config = {"extra": "allow"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ApplicantRequest) -> dict[str, Any]:
    try:
        return predict_applicant(request.model_dump())
    except PreprocessingError as exc:
        logger.warning("Invalid applicant payload: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@app.post("/decision")
def decision(request: ApplicantRequest) -> dict[str, Any]:
    """Model prediction plus business rules (for Decision Rules dashboard)."""
    try:
        payload = request.model_dump()
        prediction = predict_applicant(payload)
        return build_decision(payload, prediction=prediction)
    except PreprocessingError as exc:
        logger.warning("Invalid applicant payload: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Decision error")
        raise HTTPException(status_code=500, detail="Decision failed.") from exc


@app.get("/rules")
def rules_catalog() -> dict[str, Any]:
    return {"active_rule_count": len(list_active_rules()), "rules": list_active_rules()}


@app.get("/dashboard/summary")
def dashboard_summary(refresh: bool = False) -> dict[str, Any]:
    """Executive dashboard metrics from portfolio scoring and SHAP artifacts."""
    try:
        if refresh:
            from src.ml.portfolio_analytics import clear_portfolio_cache, get_portfolio_analytics

            clear_portfolio_cache()
            get_portfolio_analytics(force_refresh=True)
        return build_dashboard_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Dashboard summary error")
        raise HTTPException(status_code=500, detail="Dashboard summary failed.") from exc
