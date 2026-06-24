"""
Load and prepare the Home Credit training portfolio for batch scoring.

Pipeline matches notebooks/edanotebook.ipynb (feature engineering, stratified split, median imputation).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.features import SELECTED_FEATURES, build_model_dataframe
from src.utils.config import PROJECT_ROOT


def portfolio_csv_candidates() -> list[Path]:
    """Paths checked for application_train.csv (first match wins)."""
    env_path = os.environ.get("CREDIT_RISK_PORTFOLIO_CSV", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            PROJECT_ROOT / "data" / "application_train.csv",
            PROJECT_ROOT / "data" / "home_credit" / "application_train.csv",
        ]
    )
    return candidates


def resolve_portfolio_csv() -> Path | None:
    for path in portfolio_csv_candidates():
        if path.is_file():
            return path
    return None


def load_portfolio_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or resolve_portfolio_csv()
    if path is None:
        raise FileNotFoundError(
            "Portfolio CSV not found. Place application_train.csv in data/ or set CREDIT_RISK_PORTFOLIO_CSV."
        )
    raw = pd.read_csv(path)
    return build_model_dataframe(raw)


def impute_portfolio_features(model_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Median-impute features using the persisted training imputer.

    Returns imputed feature matrix for all rows and metadata about the split.
    """
    from src.data.training_imputer import impute_feature_frame, load_training_imputer

    x = model_df.drop(columns=["TARGET"])
    load_training_imputer()  # ensure artifact exists
    imputed = impute_feature_frame(x)
    meta = {
        "csv_path": str(resolve_portfolio_csv() or ""),
        "total_records": int(len(model_df)),
        "holdout_records": int(round(len(model_df) * 0.2)),
    }
    return imputed, meta
