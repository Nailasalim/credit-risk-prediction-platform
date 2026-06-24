"""
Compact schema grounding for Groq NL→SQL prompts (token-optimized).
"""

from __future__ import annotations

TABLE_NAME = "application_train"
KPI_TABLE = "portfolio_kpis"

# Curated column hints — full CSV has ~122 columns; list covers common analyst questions.
COLUMN_HINTS: dict[str, str] = {
    "TARGET": "0=repaid, 1=default (label)",
    "SK_ID_CURR": "application id",
    "NAME_CONTRACT_TYPE": "Cash loans | Revolving loans",
    "CODE_GENDER": "M | F | XNA",
    "AMT_INCOME_TOTAL": "annual income",
    "AMT_CREDIT": "loan amount",
    "AMT_ANNUITY": "installment amount",
    "AMT_GOODS_PRICE": "goods price",
    "DAYS_BIRTH": "age in days (negative); use ABS(DAYS_BIRTH)/365.25 for years",
    "DAYS_EMPLOYED": "employment days (365243 = unknown)",
    "EXT_SOURCE_1": "external score 1 (normalized)",
    "EXT_SOURCE_2": "external score 2 (normalized)",
    "EXT_SOURCE_3": "external score 3 (normalized)",
    "REGION_RATING_CLIENT": "region risk rating 1-3",
    "REGION_RATING_CLIENT_W_CITY": "region+city rating 1-3",
    "NAME_EDUCATION_TYPE": "education level",
    "NAME_INCOME_TYPE": "income type",
    "NAME_FAMILY_STATUS": "family status",
    "NAME_HOUSING_TYPE": "housing type",
    "OCCUPATION_TYPE": "occupation",
    "CNT_CHILDREN": "number of children",
    "CNT_FAM_MEMBERS": "family members",
    "FLAG_OWN_CAR": "Y/N",
    "FLAG_OWN_REALTY": "Y/N",
}

KPI_HINTS = {
    "approval_rate_pct": "batch-scored approval %",
    "decline_rate_pct": "batch-scored decline %",
    "review_rate_pct": "batch-scored review %",
    "observed_default_rate_pct": "observed default % in training data",
    "scored_records": "applications scored",
    "decision_threshold": "LightGBM decision threshold",
}


def build_schema_prompt(available_columns: list[str] | None = None) -> str:
    """
    Structured schema block for the NL→SQL prompt.

    When available_columns is provided, only documented columns present in the
    dataframe are listed to avoid hallucinated field names.
    """
    cols = available_columns or list(COLUMN_HINTS.keys())
    documented = [c for c in cols if c in COLUMN_HINTS]
    extra = [c for c in cols if c not in COLUMN_HINTS and c != "TARGET"][:40]

    lines = [
        f"TABLE {TABLE_NAME} (SQLite, Home Credit applications, ~307k rows)",
    ]
    for col in documented:
        lines.append(f"  {col}: {COLUMN_HINTS[col]}")
    if extra:
        lines.append(f"  Other columns (use exact names): {', '.join(extra[:25])}")

    lines.extend(
        [
            "",
            f"TABLE {KPI_TABLE} (batch underwriting KPIs)",
            "  metric TEXT, value REAL",
        ]
    )
    for metric, desc in KPI_HINTS.items():
        lines.append(f"  {metric}: {desc}")

    lines.extend(
        [
            "",
            "Notes:",
            "- Default rate: AVG(TARGET) or ROUND(100.0*AVG(TARGET),2) AS default_rate_pct",
            "- Filter KPIs: SELECT value FROM portfolio_kpis WHERE metric='approval_rate_pct'",
            "- SQLite dialect; one SELECT only",
        ]
    )
    return "\n".join(lines)
