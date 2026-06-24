"""
Fit and persist the training SimpleImputer (median, train-split fit).

Run from project root:
    python scripts/fit_training_imputer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_imputer import fit_and_save_training_imputer  # noqa: E402
from src.utils.config import IMPUTER_PATH  # noqa: E402


def main() -> None:
    imputer = fit_and_save_training_imputer()
    print(f"Saved imputer with {len(imputer.statistics_)} features to {IMPUTER_PATH}")


if __name__ == "__main__":
    main()
