"""
Transform raw applicant input into the 109-feature model matrix.
"""

from __future__ import annotations

import logging
from typing import Any

from src.data.feature_engineering import build_applicant_features
from src.data.training_imputer import impute_feature_frame
from src.utils.config import DAYS_EMPLOYED_UNKNOWN

logger = logging.getLogger(__name__)

RATIO_BASE_COLUMNS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
)


class PreprocessingError(Exception):
    """Raised when applicant data cannot be converted for prediction."""


def engineer_features(df):
    """Backward-compatible alias — use build_applicant_features for new code."""
    from src.data.feature_engineering import engineer_ratio_features

    return engineer_ratio_features(df)


def validate_applicant_data(data: dict[str, Any]) -> None:
    if not data:
        raise PreprocessingError("Applicant data is empty.")

    missing_ratio_inputs = [col for col in RATIO_BASE_COLUMNS if col not in data]
    if missing_ratio_inputs:
        raise PreprocessingError(
            f"Missing required fields for feature engineering: {missing_ratio_inputs}"
        )


def preprocess_applicant(data: dict[str, Any]):
    """Accept a single applicant record and return imputed features for predict_proba."""
    validate_applicant_data(data)

    try:
        if "DAYS_EMPLOYED" in data:
            employed = data["DAYS_EMPLOYED"]
            if employed == DAYS_EMPLOYED_UNKNOWN:
                data = {**data, "DAYS_EMPLOYED": None}

        features = build_applicant_features(data)
        model_input = impute_feature_frame(features)
        logger.debug(
            "Preprocessed applicant with %d features (median-imputed)", model_input.shape[1]
        )
        return model_input
    except PreprocessingError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during preprocessing")
        raise PreprocessingError(f"Failed to preprocess applicant data: {exc}") from exc
