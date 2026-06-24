"""
Portfolio-level underwriting analytics from batch model scoring.

Scores data/application_train.csv (gitignored) and caches results in
models/portfolio_scoring_snapshot.json.
"""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.data.loader import load_model
from src.data.portfolio_loader import (
    impute_portfolio_features,
    load_portfolio_dataframe,
    resolve_portfolio_csv,
)
from src.ml.predict import get_threshold
from src.utils.config import MODELS_DIR, RISK_BAND_LOW_MAX, RISK_BAND_MEDIUM_MAX

logger = logging.getLogger(__name__)

PORTFOLIO_SNAPSHOT_PATH = MODELS_DIR / "portfolio_scoring_snapshot.json"


def _vectorized_risk_bands(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    medium_cutoff = min(RISK_BAND_MEDIUM_MAX, threshold)
    low_cutoff = min(RISK_BAND_LOW_MAX, medium_cutoff)
    return np.select(
        [probabilities < low_cutoff, probabilities < medium_cutoff],
        ["LOW", "MEDIUM"],
        default="HIGH",
    )


def _vectorized_model_decisions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(probabilities >= threshold, "REJECT", "APPROVE")


def _vectorized_recommendations(bands: np.ndarray, model_decisions: np.ndarray) -> np.ndarray:
    """Same mapping as recommendation_from_band (band policy; rules apply per-applicant only)."""
    return np.select(
        [bands == "MEDIUM", (bands == "HIGH") | (model_decisions == "REJECT")],
        ["REVIEW", "DECLINE"],
        default="APPROVE",
    )


def compute_portfolio_from_dataframe(model_df) -> dict[str, Any]:
    """Score the full training portfolio and aggregate underwriting metrics."""
    started = time.perf_counter()
    imputed, meta = impute_portfolio_features(model_df)
    model = load_model()
    threshold = get_threshold()

    probabilities = model.predict_proba(imputed)[:, 1].astype(np.float64)
    bands = _vectorized_risk_bands(probabilities, threshold)
    model_decisions = _vectorized_model_decisions(probabilities, threshold)
    recommendations = _vectorized_recommendations(bands, model_decisions)

    total = int(len(probabilities))
    band_counts = {
        "LOW": int(np.sum(bands == "LOW")),
        "MEDIUM": int(np.sum(bands == "MEDIUM")),
        "HIGH": int(np.sum(bands == "HIGH")),
    }

    prob_bucket_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.67), (0.67, 1.0000001)]
    prob_bucket_labels = ["0–20%", "20–40%", "40–67%", "67%+"]
    probability_bucket_distribution = []
    for (lo, hi), label in zip(prob_bucket_edges, prob_bucket_labels):
        mask = (probabilities >= lo) & (probabilities < hi)
        count = int(np.sum(mask))
        probability_bucket_distribution.append(
            {
                "bucket": label,
                "count": count,
                "pct": round(count / total * 100, 1) if total else 0.0,
            }
        )
    rec_counts = {
        "APPROVE": int(np.sum(recommendations == "APPROVE")),
        "REVIEW": int(np.sum(recommendations == "REVIEW")),
        "DECLINE": int(np.sum(recommendations == "DECLINE")),
    }

    observed_default_rate = round(float(model_df["TARGET"].mean()) * 100, 1)
    elapsed_sec = round(time.perf_counter() - started, 2)

    return {
        "source": "batch_model_scoring",
        "csv_path": meta.get("csv_path", ""),
        "scored_records": total,
        "portfolio_total": total,
        "observed_default_rate_pct": observed_default_rate,
        "approval_rate_pct": round(rec_counts["APPROVE"] / total * 100, 1),
        "review_rate_pct": round(rec_counts["REVIEW"] / total * 100, 1),
        "decline_rate_pct": round(rec_counts["DECLINE"] / total * 100, 1),
        "high_risk_count": band_counts["HIGH"],
        "high_risk_pct": round(band_counts["HIGH"] / total * 100, 1),
        "probability_bucket_distribution": probability_bucket_distribution,
        "risk_band_distribution": [
            {
                "band": "LOW",
                "count": band_counts["LOW"],
                "pct": round(band_counts["LOW"] / total * 100, 1),
            },
            {
                "band": "MEDIUM",
                "count": band_counts["MEDIUM"],
                "pct": round(band_counts["MEDIUM"] / total * 100, 1),
            },
            {
                "band": "HIGH",
                "count": band_counts["HIGH"],
                "pct": round(band_counts["HIGH"] / total * 100, 1),
            },
        ],
        "recommendation_distribution": [
            {"action": action, "count": count, "pct": round(count / total * 100, 1)}
            for action, count in rec_counts.items()
        ],
        "threshold": threshold,
        "scoring_runtime_seconds": elapsed_sec,
    }


def save_portfolio_snapshot(payload: dict[str, Any], path: Path = PORTFOLIO_SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_portfolio_snapshot(path: Path = PORTFOLIO_SNAPSHOT_PATH) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _snapshot_is_fresh(csv_path: Path, snapshot_path: Path = PORTFOLIO_SNAPSHOT_PATH) -> bool:
    if not snapshot_path.is_file():
        return False
    try:
        return snapshot_path.stat().st_mtime >= csv_path.stat().st_mtime
    except OSError:
        return False


def build_portfolio_snapshot_from_csv(csv_path: Path | None = None) -> dict[str, Any]:
    path = csv_path or resolve_portfolio_csv()
    if path is None:
        raise FileNotFoundError("application_train.csv not found under data/")
    model_df = load_portfolio_dataframe(path)
    payload = compute_portfolio_from_dataframe(model_df)
    save_portfolio_snapshot(payload)
    return payload


@lru_cache(maxsize=1)
def get_portfolio_analytics(*, force_refresh: bool = False) -> dict[str, Any]:
    """
    Return portfolio underwriting analytics from CSV scoring (cached snapshot).

    Recomputes when CSV is newer than the snapshot or when force_refresh=True.
    """
    csv_path = resolve_portfolio_csv()
    if csv_path is None:
        raise FileNotFoundError(
            "Portfolio CSV not found. Expected data/application_train.csv (gitignored)."
        )

    if not force_refresh and _snapshot_is_fresh(csv_path):
        snapshot = load_portfolio_snapshot()
        if snapshot is not None:
            if "probability_bucket_distribution" not in snapshot:
                logger.info("Refreshing portfolio snapshot (missing probability buckets)")
                return build_portfolio_snapshot_from_csv(csv_path)
            return snapshot

    logger.info("Scoring portfolio from %s", csv_path)
    return build_portfolio_snapshot_from_csv(csv_path)


def clear_portfolio_cache() -> None:
    get_portfolio_analytics.cache_clear()
