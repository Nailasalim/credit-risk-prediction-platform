"""
Full Home Credit feature engineering pipeline (109 model features).

Feature set:
  - 106 columns from application_train (label-encoded categoricals)
  - Minus 14 redundant FLAG_DOCUMENT_* fields (keeps FLAG_DOCUMENT_3, 17, 18, 19, 20, 21)
  - Plus 3 engineered ratios (INCOME_CREDIT_RATIO, ANNUITY_INCOME_RATIO, CREDIT_GOODS_RATIO)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.config import DAYS_EMPLOYED_UNKNOWN, ENCODERS_PATH, FEATURE_NAMES_PATH

logger = logging.getLogger(__name__)

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

# Drop 14 low-signal document flags; retain FLAG_DOCUMENT_3 (used in Phase 1 rules/UI).
DOCUMENT_FLAGS_TO_DROP: tuple[str, ...] = tuple(
    f"FLAG_DOCUMENT_{i}"
    for i in (2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
)

RATIO_FEATURES: tuple[str, ...] = (
    "INCOME_CREDIT_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
)

EXPECTED_FEATURE_COUNT = 109


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if df[col].dtype == "object"]


def fit_label_encoders(frame: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Fit label encoders on training split categoricals."""
    encoders: dict[str, LabelEncoder] = {}
    for col in _categorical_columns(frame):
        encoder = LabelEncoder()
        values = frame[col].astype(str).fillna("__MISSING__")
        encoder.fit(values)
        encoders[col] = encoder
    logger.info("Fitted %d label encoders", len(encoders))
    return encoders


def apply_label_encoders(frame: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> pd.DataFrame:
    """Transform categoricals; unseen labels map to -1."""
    out = frame.copy()
    for col, encoder in encoders.items():
        if col not in out.columns:
            out[col] = -1
            continue
        mapping = {label: int(idx) for idx, label in enumerate(encoder.classes_)}
        values = out[col].astype(str).fillna("__MISSING__")
        out[col] = values.map(lambda value: mapping.get(value, -1)).astype(float)
    return out


def _prepare_raw_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop IDs, target, and redundant document flags from raw Home Credit rows."""
    frame = raw.copy()
    drop_cols = [c for c in (ID_COLUMN, TARGET_COLUMN, *DOCUMENT_FLAGS_TO_DROP) if c in frame.columns]
    return frame.drop(columns=drop_cols, errors="ignore")


def engineer_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add financial ratio features."""
    out = frame.copy()
    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_UNKNOWN, np.nan)
    out["INCOME_CREDIT_RATIO"] = _safe_ratio(out["AMT_INCOME_TOTAL"], out["AMT_CREDIT"])
    out["ANNUITY_INCOME_RATIO"] = _safe_ratio(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["CREDIT_GOODS_RATIO"] = _safe_ratio(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
    return out


def build_feature_matrix(
    raw: pd.DataFrame,
    *,
    encoders: dict[str, LabelEncoder] | None = None,
    fit_encoders: bool = False,
) -> pd.DataFrame:
    """
    Transform raw application rows into the 109-feature model matrix.

    When fit_encoders=True, encoders are fit on the provided frame (training split only).
    """
    if TARGET_COLUMN in raw.columns:
        y = raw[TARGET_COLUMN]
    else:
        y = None

    base = _prepare_raw_features(raw)
    if fit_encoders:
        encoders = fit_label_encoders(base)
    if encoders is None:
        encoders = load_label_encoders()

    encoded = apply_label_encoders(base, encoders)
    featured = engineer_ratio_features(encoded)

    if y is not None:
        featured[TARGET_COLUMN] = y.values

    feature_cols = [c for c in featured.columns if c != TARGET_COLUMN]
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(feature_cols)}. "
            f"Check DOCUMENT_FLAGS_TO_DROP and ratio features."
        )

    return featured


def feature_column_names(frame: pd.DataFrame) -> list[str]:
    """Ordered feature names (excluding TARGET)."""
    return [c for c in frame.columns if c != TARGET_COLUMN]


def save_label_encoders(encoders: dict[str, LabelEncoder], path: Path = ENCODERS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, path)
    logger.info("Saved label encoders to %s", path)


def load_label_encoders(path: Path = ENCODERS_PATH) -> dict[str, LabelEncoder]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Label encoders not found at {path}. Run scripts/train_model.py first."
        )
    return joblib.load(path)


def save_feature_names(names: list[str], path: Path = FEATURE_NAMES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(names, file, indent=2)
    logger.info("Saved %d feature names to %s", len(names), path)


def build_model_dataframe(
    raw: pd.DataFrame,
    *,
    encoders: dict[str, LabelEncoder] | None = None,
) -> pd.DataFrame:
    """Portfolio/training helper — returns features + TARGET."""
    if encoders is None:
        try:
            encoders = load_label_encoders()
        except FileNotFoundError:
            encoders = None
    return build_feature_matrix(raw, encoders=encoders, fit_encoders=encoders is None)


def build_applicant_features(
    applicant: dict[str, Any],
    *,
    encoders: dict[str, LabelEncoder] | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build one applicant row for inference.

    Missing raw fields are left as NaN and handled by the training imputer.
    """
    if encoders is None:
        encoders = load_label_encoders()
    if feature_names is None:
        from src.data.loader import load_feature_names

        feature_names = load_feature_names()

    row = pd.DataFrame([{**{name: np.nan for name in feature_names}, **applicant}])
    # Re-derive ratios from financial inputs when available.
    row = engineer_ratio_features(row)

    categorical_cols = [col for col in encoders if col in row.columns]
    if categorical_cols:
        encoded = apply_label_encoders(row[categorical_cols], encoders)
        row[categorical_cols] = encoded[categorical_cols]

    return row[feature_names]
