"""
Transform raw applicant input into the feature matrix used at training time.

Engineered ratios match the formulas from Phase 1 feature engineering.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import load_feature_names
from src.data.training_imputer import impute_feature_frame
from src.utils.config import DAYS_EMPLOYED_UNKNOWN

logger = logging.getLogger(__name__)

# Base columns required to compute ratio features (plus any other model inputs).
RATIO_BASE_COLUMNS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
)


class PreprocessingError(Exception):
    """Raised when applicant data cannot be converted for prediction."""


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series; invalid divisions become NaN (same as training with missing values)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio features using the same formulas as the training notebook.

    INCOME_CREDIT_RATIO  = AMT_INCOME_TOTAL / AMT_CREDIT
    ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL
    CREDIT_GOODS_RATIO   = AMT_CREDIT / AMT_GOODS_PRICE
    """
    df = df.copy()

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_UNKNOWN, np.nan)

    df["INCOME_CREDIT_RATIO"] = _safe_ratio(df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"])
    df["ANNUITY_INCOME_RATIO"] = _safe_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_GOODS_RATIO"] = _safe_ratio(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])

    return df


def validate_applicant_data(data: dict[str, Any]) -> None:
    """Ensure required fields are present before feature engineering."""
    if not data:
        raise PreprocessingError("Applicant data is empty.")

    missing_ratio_inputs = [col for col in RATIO_BASE_COLUMNS if col not in data]
    if missing_ratio_inputs:
        raise PreprocessingError(
            f"Missing required fields for feature engineering: {missing_ratio_inputs}"
        )


def preprocess_applicant(data: dict[str, Any]) -> pd.DataFrame:
    """
    Accept a single applicant record and return a one-row DataFrame
    with columns ordered for model.predict_proba.
    """
    validate_applicant_data(data)

    try:
        feature_names = load_feature_names()
        df = pd.DataFrame([data])
        df = engineer_features(df)

        missing_features = [name for name in feature_names if name not in df.columns]
        if missing_features:
            raise PreprocessingError(
                f"Missing model features after preprocessing: {missing_features}"
            )

        engineered = df[feature_names]
        model_input = impute_feature_frame(engineered)
        logger.debug(
            "Preprocessed applicant with %d features (median-imputed)", len(feature_names)
        )
        return model_input
    except PreprocessingError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during preprocessing")
        raise PreprocessingError(f"Failed to preprocess applicant data: {exc}") from exc
