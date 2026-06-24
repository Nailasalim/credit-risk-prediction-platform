"""
Training-aligned median imputation (SimpleImputer fit on the 80% train split).

The fitted imputer is persisted to models/imputer.pkl so single-applicant inference
matches the notebook pipeline without re-reading the full portfolio CSV.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.data.loader import load_feature_names
from src.data.features import build_model_dataframe
from src.data.portfolio_loader import resolve_portfolio_csv
from src.utils.config import (
    IMPUTER_PATH,
    TRAINING_SPLIT_RANDOM_STATE,
    TRAINING_SPLIT_TEST_SIZE,
)

logger = logging.getLogger(__name__)


def fit_training_imputer(model_df: pd.DataFrame) -> SimpleImputer:
    """Fit median imputer on X_train (same stratified split as training notebook)."""
    x = model_df.drop(columns=["TARGET"])
    y = model_df["TARGET"]
    x_train, _, _, _ = train_test_split(
        x,
        y,
        test_size=TRAINING_SPLIT_TEST_SIZE,
        random_state=TRAINING_SPLIT_RANDOM_STATE,
        stratify=y,
    )
    imputer = SimpleImputer(strategy="median")
    imputer.fit(x_train)
    logger.info("Fitted training imputer on %d train rows", len(x_train))
    return imputer


def save_training_imputer(imputer: SimpleImputer, path=IMPUTER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, path)
    logger.info("Saved training imputer to %s", path)


def _load_portfolio_model_df() -> pd.DataFrame:
    csv_path = resolve_portfolio_csv()
    if csv_path is None:
        raise FileNotFoundError(
            "application_train.csv required to fit imputer. "
            "Run scripts/fit_training_imputer.py or place CSV in data/."
        )
    import pandas as pd

    return build_model_dataframe(pd.read_csv(csv_path))


def fit_and_save_training_imputer(model_df: pd.DataFrame | None = None) -> SimpleImputer:
    if model_df is None:
        model_df = _load_portfolio_model_df()
    imputer = fit_training_imputer(model_df)
    save_training_imputer(imputer)
    return imputer


@lru_cache(maxsize=1)
def load_training_imputer() -> SimpleImputer:
    """Load persisted imputer or fit from portfolio CSV when artifact is missing."""
    if IMPUTER_PATH.is_file():
        logger.info("Loading training imputer from %s", IMPUTER_PATH)
        return joblib.load(IMPUTER_PATH)

    logger.warning(
        "Imputer artifact missing at %s; fitting from portfolio CSV", IMPUTER_PATH
    )
    return fit_and_save_training_imputer()


def impute_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Apply training median imputation; preserves column order and index."""
    imputer = load_training_imputer()
    imputed = imputer.transform(features)
    return pd.DataFrame(imputed, columns=features.columns, index=features.index)


def training_imputer_statistics() -> pd.Series:
    """Median fill values per feature (for audits)."""
    imputer = load_training_imputer()
    names = list(getattr(imputer, "feature_names_in_", None) or load_feature_names())
    return pd.Series(imputer.statistics_, index=names)
