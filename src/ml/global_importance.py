"""
Portfolio-level feature importance from saved SHAP values (or model gain fallback).

Shared by the Executive Dashboard and Explainability page.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.loader import load_feature_names, load_model
from src.utils.config import SHAP_VALUES_PATH


@lru_cache(maxsize=1)
def load_global_importance() -> pd.DataFrame:
    """
    Rank features by mean |SHAP| with signed display impact (median sign × mean |SHAP|).

    Falls back to normalized LightGBM gain when shap_values.npy is missing.
    """
    names = load_feature_names()

    if SHAP_VALUES_PATH.is_file():
        shap_matrix = np.load(SHAP_VALUES_PATH)
        if shap_matrix.ndim != 2 or shap_matrix.shape[1] != len(names):
            raise ValueError(
                f"Expected shap_values shape (n_samples, {len(names)}), got {shap_matrix.shape}"
            )
        mean_abs = np.abs(shap_matrix).mean(axis=0)
        median_signed = np.median(shap_matrix, axis=0)
        direction = np.sign(median_signed)
        direction = np.where(direction == 0, np.sign(shap_matrix.mean(axis=0)), direction)
        direction = np.where(direction == 0, 1.0, direction)
        display_impact = direction * mean_abs
        source = "global_shap"
        sample_count = int(shap_matrix.shape[0])
    else:
        model = load_model()
        raw = np.asarray(model.feature_importances_, dtype=float)
        mean_abs = raw / raw.sum()
        display_impact = mean_abs
        source = "model_gain"
        sample_count = 0

    frame = pd.DataFrame(
        {
            "feature": names,
            "importance": mean_abs,
            "display_impact": display_impact,
            "source": source,
            "sample_count": sample_count,
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def top_global_features(limit: int = 8) -> pd.DataFrame:
    """Return the top N globally important features."""
    return load_global_importance().head(limit).copy()
