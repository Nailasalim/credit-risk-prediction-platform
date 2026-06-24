"""
Central configuration for model artifacts and inference settings.

All paths are resolved relative to the project root so the API can run
from different working directories.
"""

from pathlib import Path

# Project root: credit_risk_prediction/ (parent of src/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# --- Artifact paths ---
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "model.pkl"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"
FEATURE_NAMES_PATH: Path = MODELS_DIR / "feature_names.json"
IMPUTER_PATH: Path = MODELS_DIR / "imputer.pkl"
SHAP_VALUES_PATH: Path = MODELS_DIR / "shap_values.npy"

# Must match notebooks/edanotebook.ipynb train/test split for imputer fitting
TRAINING_SPLIT_RANDOM_STATE: int = 42
TRAINING_SPLIT_TEST_SIZE: float = 0.2

# --- Inference settings ---
# Risk bands use fixed cutoffs; decision uses the tuned threshold from metrics.json.
RISK_BAND_LOW_MAX: float = 0.40
RISK_BAND_MEDIUM_MAX: float = 0.67

# Sentinel value in Home Credit data meaning "unknown employment duration"
DAYS_EMPLOYED_UNKNOWN: int = 365243

# Home Credit training set size (Phase 1) — used for rule coverage metrics
TRAINING_PORTFOLIO_SIZE: int = 307_511
