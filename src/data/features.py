"""
Shared feature list and engineering for training and inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import DAYS_EMPLOYED_UNKNOWN

SELECTED_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "INCOME_CREDIT_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_ID_PUBLISH",
    "REG_CITY_NOT_WORK_CITY",
    "REG_CITY_NOT_LIVE_CITY",
    "FLAG_EMP_PHONE",
    "FLAG_DOCUMENT_3",
    "OWN_CAR_AGE",
]


def build_model_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer ratios and select model features (notebook-aligned)."""
    df = raw.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_UNKNOWN, np.nan)
    df["INCOME_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    return df[SELECTED_FEATURES + ["TARGET"]]
