"""
Training-set portfolio statistics (Phase 1).

Values are loaded from documents/dataset_summary.json, aligned with phase1_findings.md.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT

DATASET_SUMMARY_PATH = PROJECT_ROOT / "documents" / "dataset_summary.json"


@lru_cache(maxsize=1)
def load_dataset_summary() -> dict[str, Any]:
    with DATASET_SUMMARY_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def dataset_summary_for_api() -> dict[str, Any]:
    """Return dataset counts and imbalance for dashboard consumers."""
    raw = load_dataset_summary()
    total = int(raw["total_records"])
    default_count = int(raw["default_count"])
    non_default_count = int(raw["non_default_count"])
    default_pct = float(raw["default_rate_pct"])
    return {
        "source": raw.get("source", ""),
        "total_records": total,
        "default_count": default_count,
        "non_default_count": non_default_count,
        "default_rate_pct": default_pct,
        "non_default_rate_pct": float(raw["non_default_rate_pct"]),
        "class_imbalance_ratio": round(non_default_count / default_count, 1) if default_count else None,
    }
