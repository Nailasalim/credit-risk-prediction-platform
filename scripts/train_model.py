"""
Train LightGBM on 109 engineered Home Credit features with scale_pos_weight.

Run from project root:
    python scripts/train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import (  # noqa: E402
    TARGET_COLUMN,
    _prepare_raw_features,
    build_feature_matrix,
    feature_column_names,
    fit_label_encoders,
    save_feature_names,
    save_label_encoders,
)
from src.data.portfolio_loader import resolve_portfolio_csv  # noqa: E402
from src.utils.config import (  # noqa: E402
    ENCODERS_PATH,
    FEATURE_NAMES_PATH,
    IMPUTER_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SHAP_VALUES_PATH,
    TRAINING_SPLIT_RANDOM_STATE,
    TRAINING_SPLIT_TEST_SIZE,
)

SHAP_SAMPLE_SIZE = 5000


def _best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Pick threshold that maximizes F1 on validation data."""
    precisions, recalls, thresholds = [], [], []
    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (probabilities >= threshold).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
    f1_scores = [
        2 * p * r / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(precisions, recalls, strict=True)
    ]
    best_idx = int(np.argmax(f1_scores))
    return float(round(np.arange(0.05, 0.95, 0.01)[best_idx], 2))


def main() -> None:
    csv_path = resolve_portfolio_csv()
    if csv_path is None:
        raise FileNotFoundError("Place application_train.csv in data/ before training.")

    print(f"Loading {csv_path} ...")
    raw = pd.read_csv(csv_path)
    y = raw[TARGET_COLUMN]

    x_train_raw, x_holdout_raw, y_train, y_holdout = train_test_split(
        raw,
        y,
        test_size=TRAINING_SPLIT_TEST_SIZE,
        random_state=TRAINING_SPLIT_RANDOM_STATE,
        stratify=y,
    )

    encoders = fit_label_encoders(_prepare_raw_features(x_train_raw))
    save_label_encoders(encoders, ENCODERS_PATH)

    train_matrix = build_feature_matrix(x_train_raw, encoders=encoders, fit_encoders=False)
    holdout_matrix = build_feature_matrix(x_holdout_raw, encoders=encoders, fit_encoders=False)

    feature_names = feature_column_names(train_matrix)
    save_feature_names(feature_names, FEATURE_NAMES_PATH)
    print(f"Feature count: {len(feature_names)}")

    x_train = train_matrix[feature_names]
    x_holdout = holdout_matrix[feature_names]

    imputer = SimpleImputer(strategy="median")
    x_train_imp = pd.DataFrame(imputer.fit_transform(x_train), columns=feature_names)
    x_holdout_imp = pd.DataFrame(imputer.transform(x_holdout), columns=feature_names)
    IMPUTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, IMPUTER_PATH)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=TRAINING_SPLIT_RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    print("Training LightGBM ...")
    model.fit(x_train_imp, y_train)

    probabilities = model.predict_proba(x_holdout_imp)[:, 1]
    threshold = _best_threshold(y_holdout.to_numpy(), probabilities)

    preds = (probabilities >= threshold).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_holdout, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_holdout, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_holdout, preds)), 4),
        "precision": round(float(precision_score(y_holdout, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_holdout, preds, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_holdout, preds, zero_division=0)), 4),
        "threshold": threshold,
        "feature_count": len(feature_names),
        "scale_pos_weight": round(scale_pos_weight, 2),
    }

    joblib.dump(model, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Holdout metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    sample_n = min(SHAP_SAMPLE_SIZE, len(x_holdout_imp))
    rng = np.random.default_rng(TRAINING_SPLIT_RANDOM_STATE)
    sample_idx = rng.choice(len(x_holdout_imp), size=sample_n, replace=False)
    x_shap = x_holdout_imp.iloc[sample_idx].to_numpy()

    print(f"Computing SHAP values on {sample_n} holdout rows ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    np.save(SHAP_VALUES_PATH, np.asarray(shap_values))

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved SHAP -> {SHAP_VALUES_PATH}")


if __name__ == "__main__":
    main()
