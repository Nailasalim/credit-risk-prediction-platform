"""
Human-readable rule explanations generated from structured conditions.

Deterministic — no LLM, no hardcoded per-rule narratives.
"""

from __future__ import annotations

from typing import Any, Literal

Operator = Literal["<", ">", "<=", ">=", "==", "!="]

FEATURE_LABELS: dict[str, str] = {
    "EXT_SOURCE_1": "External Credit Score 1",
    "EXT_SOURCE_2": "External Credit Score 2",
    "EXT_SOURCE_3": "External Credit Score 3",
    "AMT_INCOME_TOTAL": "Annual Income",
    "AMT_CREDIT": "Requested Loan Amount",
    "AMT_ANNUITY": "Monthly EMI",
    "AMT_GOODS_PRICE": "Goods Price",
    "ANNUITY_INCOME_RATIO": "EMI-to-Income Ratio",
    "INCOME_CREDIT_RATIO": "Income-to-Loan Ratio",
    "CREDIT_GOODS_RATIO": "Loan-to-Goods Ratio",
    "DAYS_EMPLOYED": "Employment Tenure",
    "DAYS_BIRTH": "Applicant Age",
    "REGION_RATING_CLIENT": "Region Risk Rating",
    "REGION_RATING_CLIENT_W_CITY": "Region & City Rating",
    "DAYS_LAST_PHONE_CHANGE": "Phone Number Stability",
    "DAYS_ID_PUBLISH": "ID Document Tenure",
    "REG_CITY_NOT_WORK_CITY": "Registered vs Work Address",
    "REG_CITY_NOT_LIVE_CITY": "Registered vs Residence",
    "FLAG_EMP_PHONE": "Employer Phone on File",
    "FLAG_DOCUMENT_3": "ID Document Verified",
    "OWN_CAR_AGE": "Vehicle Age",
}

CURRENCY_FEATURES = frozenset(
    {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}
)
SCORE_FEATURES = frozenset({"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"})

ACTION_LABELS = {
    "DECLINE": "Decline",
    "REVIEW": "Manual review",
    "APPROVE": "Approve",
}

RISK_BAND_LABELS = {
    "HIGH": "High Risk",
    "MEDIUM": "Medium Risk",
    "LOW": "Low Risk",
}

WHY_RULE_EXISTS = (
    "Derived from LightGBM decision paths observed across the training portfolio."
)


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _format_threshold(feature: str, threshold: float | int, unit: str | None) -> str:
    if unit == "years":
        years = float(threshold)
        if years == 1.0:
            return "1 year"
        return f"{years:g} years"
    if feature in CURRENCY_FEATURES or unit == "currency":
        return f"₹{float(threshold):,.0f}"
    if feature in SCORE_FEATURES or unit == "ratio":
        return f"{float(threshold):.2f}"
    if isinstance(threshold, float) and threshold == int(threshold):
        return str(int(threshold))
    return str(threshold)


def format_condition_compact(condition: dict[str, Any]) -> str:
    """Underwriting-style compact expression, e.g. ``External Credit Score 2 < 0.35``."""
    feature = str(condition["feature"])
    operator = str(condition["operator"])
    threshold = condition["threshold"]
    unit = condition.get("unit")
    label = feature_label(feature)
    value = _format_threshold(feature, threshold, unit)
    return f"{label} {operator} {value}"


def humanize_condition(condition: dict[str, Any]) -> str:
    """Convert one structured condition to readable English (full sentence form)."""
    feature = str(condition["feature"])
    operator = str(condition["operator"])
    threshold = condition["threshold"]
    unit = condition.get("unit")
    label = feature_label(feature)
    value = _format_threshold(feature, threshold, unit)

    if operator == "<":
        return f"{label} is below {value}"
    if operator == ">":
        return f"{label} exceeds {value}"
    if operator == "<=":
        return f"{label} is at most {value}"
    if operator == ">=":
        return f"{label} is at least {value}"
    if operator == "==":
        return f"{label} equals {value}"
    if operator == "!=":
        return f"{label} does not equal {value}"
    return f"{label} {operator} {value}"


def _join_clauses(clauses: list[str]) -> str:
    if not clauses:
        return "the configured conditions are met"
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


_LOW_OPS = frozenset({"<", "<="})
_HIGH_OPS = frozenset({">", ">="})


def _condition_rationale_phrase(condition: dict[str, Any]) -> str | None:
    """Compact risk phrase for one condition (used in rule rationale summary)."""
    feature = str(condition["feature"])
    operator = str(condition["operator"])

    if feature in SCORE_FEATURES:
        return None  # handled via bureau aggregation
    if feature == "AMT_CREDIT" and operator in _HIGH_OPS:
        return "high borrowing exposure"
    if feature == "AMT_INCOME_TOTAL" and operator in _LOW_OPS:
        return "low income"
    if feature == "DAYS_EMPLOYED" and operator in _LOW_OPS:
        return "short employment tenure"
    if feature == "DAYS_EMPLOYED" and operator in _HIGH_OPS:
        return "stable employment"
    if feature == "ANNUITY_INCOME_RATIO" and operator in _HIGH_OPS:
        return "elevated payment burden"
    if feature == "ANNUITY_INCOME_RATIO" and operator in _LOW_OPS:
        return "affordable payment load"
    if feature == "INCOME_CREDIT_RATIO" and operator in _LOW_OPS:
        return "thin income coverage"
    if feature == "REGION_RATING_CLIENT":
        return "weaker region risk"
    return None


def generate_rule_rationale(rule: dict[str, Any]) -> str:
    """Short risk summary phrase derived from condition semantics."""
    structured = rule.get("conditions") or []
    phrases: list[str] = []

    bureau_low_count = sum(
        1
        for c in structured
        if str(c["feature"]) in SCORE_FEATURES and str(c["operator"]) in _LOW_OPS
    )
    bureau_high_count = sum(
        1
        for c in structured
        if str(c["feature"]) in SCORE_FEATURES and str(c["operator"]) in _HIGH_OPS
    )

    if bureau_low_count >= 2:
        phrases.append("multiple weak external credit indicators")
    elif bureau_low_count == 1:
        phrases.append("low bureau score")

    if bureau_high_count >= 2:
        phrases.append("strong bureau performance")
    elif bureau_high_count == 1:
        phrases.append("strong bureau score")

    for condition in structured:
        phrase = _condition_rationale_phrase(condition)
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    if not phrases:
        return "Applicant profile match"

    summary = " and ".join(phrases)
    return summary[0].upper() + summary[1:]


def generate_rule_explanation(rule: dict[str, Any]) -> str:
    """
    Build a multi-sentence explanation from structured rule conditions.

    Parameters
    ----------
    rule:
        Dict with keys ``conditions`` (list of feature/operator/threshold),
        ``risk_band``, and ``action``.
    """
    structured = rule.get("conditions") or []
    clauses = [humanize_condition(c) for c in structured]
    trigger_text = _join_clauses(clauses)

    band = RISK_BAND_LABELS.get(str(rule.get("risk_band", "")).upper(), "this risk")
    action = ACTION_LABELS.get(str(rule.get("action", "")).upper(), "Review")

    return (
        f"This rule is triggered when {trigger_text}. "
        f"Applicants matching these conditions historically fall into the {band} segment. "
        f"Recommended action: {action}."
    )
