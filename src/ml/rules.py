"""
Business-readable decision rules for the Decision Rules dashboard.

Rules are defined as structured conditions; explanations are generated
programmatically from those conditions (see rule_explain.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

import pandas as pd

from src.data.preprocessor import engineer_features
from src.ml.predict import predict_applicant
from src.ml.rule_explain import (
    WHY_RULE_EXISTS,
    format_condition_compact,
    generate_rule_explanation,
    generate_rule_rationale,
)
from src.utils.config import DAYS_EMPLOYED_UNKNOWN, TRAINING_PORTFOLIO_SIZE

logger = logging.getLogger(__name__)

Recommendation = Literal["APPROVE", "REVIEW", "DECLINE"]
RiskBandLabel = Literal["LOW", "MEDIUM", "HIGH"]
Operator = Literal["<", ">", "<=", ">=", "==", "!="]


@dataclass(frozen=True)
class RuleCondition:
    feature: str
    operator: Operator
    threshold: float | int
    unit: str | None = None  # years | currency | ratio | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature": self.feature,
            "operator": self.operator,
            "threshold": self.threshold,
        }
        if self.unit is not None:
            payload["unit"] = self.unit
        return payload


@dataclass(frozen=True)
class RuleDefinition:
    """Catalog entry with structured conditions and mined portfolio metrics."""

    rule_id: str
    risk_band: RiskBandLabel
    title: str
    conditions: tuple[RuleCondition, ...]
    action: Recommendation
    confidence: float  # percentage 0–100 from rule mining
    support: int
    lift: float
    precision: float
    recall: float


def _num(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _employment_years(days_employed: float) -> float | None:
    if pd.isna(days_employed) or days_employed == DAYS_EMPLOYED_UNKNOWN:
        return None
    return abs(days_employed) / 365.25


def _resolve_feature_value(context: dict[str, Any], condition: RuleCondition) -> float:
    raw = _num(context.get(condition.feature))
    if condition.unit == "years" and condition.feature == "DAYS_EMPLOYED":
        years = _employment_years(raw)
        return float("nan") if years is None else years
    return raw


def _compare(actual: float, operator: Operator, threshold: float | int) -> bool:
    if pd.isna(actual):
        return False
    target = float(threshold)
    if operator == "<":
        return actual < target
    if operator == ">":
        return actual > target
    if operator == "<=":
        return actual <= target
    if operator == ">=":
        return actual >= target
    if operator == "==":
        return actual == target
    if operator == "!=":
        return actual != target
    return False


def build_rule_evaluator(conditions: tuple[RuleCondition, ...]) -> Callable[[dict[str, Any]], bool]:
    """Create a predicate that tests all structured conditions against applicant context."""

    def evaluate(context: dict[str, Any]) -> bool:
        return all(
            _compare(_resolve_feature_value(context, condition), condition.operator, condition.threshold)
            for condition in conditions
        )

    return evaluate


# --- Rule catalog (structured conditions + mined metrics) ---
ACTIVE_RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition(
        rule_id="R001",
        risk_band="HIGH",
        title="Weak bureau score + high credit exposure",
        conditions=(
            RuleCondition("EXT_SOURCE_2", "<", 0.35, "ratio"),
            RuleCondition("AMT_CREDIT", ">", 500_000, "currency"),
        ),
        action="DECLINE",
        confidence=91.0,
        support=12_410,
        lift=2.4,
        precision=0.88,
        recall=0.71,
    ),
    RuleDefinition(
        rule_id="R002",
        risk_band="HIGH",
        title="Short tenure + low income",
        conditions=(
            RuleCondition("DAYS_EMPLOYED", "<", 1.0, "years"),
            RuleCondition("AMT_INCOME_TOTAL", "<", 60_000, "currency"),
        ),
        action="DECLINE",
        confidence=88.0,
        support=8_932,
        lift=2.1,
        precision=0.85,
        recall=0.68,
    ),
    RuleDefinition(
        rule_id="R003",
        risk_band="HIGH",
        title="Low external score bundle",
        conditions=(
            RuleCondition("EXT_SOURCE_1", "<", 0.30, "ratio"),
            RuleCondition("EXT_SOURCE_3", "<", 0.40, "ratio"),
        ),
        action="DECLINE",
        confidence=86.0,
        support=7_105,
        lift=2.0,
        precision=0.83,
        recall=0.66,
    ),
    RuleDefinition(
        rule_id="R004",
        risk_band="MEDIUM",
        title="Annuity stress + weaker region rating",
        conditions=(
            RuleCondition("ANNUITY_INCOME_RATIO", ">", 0.30, "ratio"),
            RuleCondition("REGION_RATING_CLIENT", "==", 3),
        ),
        action="REVIEW",
        confidence=82.0,
        support=6_218,
        lift=1.7,
        precision=0.79,
        recall=0.74,
    ),
    RuleDefinition(
        rule_id="R005",
        risk_band="MEDIUM",
        title="Elevated payment burden",
        conditions=(
            RuleCondition("ANNUITY_INCOME_RATIO", ">", 0.25, "ratio"),
            RuleCondition("INCOME_CREDIT_RATIO", "<", 0.20, "ratio"),
        ),
        action="REVIEW",
        confidence=80.0,
        support=5_440,
        lift=1.5,
        precision=0.76,
        recall=0.70,
    ),
    RuleDefinition(
        rule_id="R006",
        risk_band="LOW",
        title="Strong bureau scores — fast-track approve",
        conditions=(
            RuleCondition("EXT_SOURCE_2", ">=", 0.70, "ratio"),
            RuleCondition("EXT_SOURCE_3", ">=", 0.65, "ratio"),
        ),
        action="APPROVE",
        confidence=94.0,
        support=21_880,
        lift=0.3,
        precision=0.96,
        recall=0.81,
    ),
    RuleDefinition(
        rule_id="R007",
        risk_band="LOW",
        title="Stable employment + affordable annuity",
        conditions=(
            RuleCondition("DAYS_EMPLOYED", ">=", 5.0, "years"),
            RuleCondition("ANNUITY_INCOME_RATIO", "<", 0.25, "ratio"),
        ),
        action="APPROVE",
        confidence=92.0,
        support=18_440,
        lift=0.4,
        precision=0.94,
        recall=0.78,
    ),
)

_RULE_EVALUATORS: dict[str, Callable[[dict[str, Any]], bool]] = {
    rule.rule_id: build_rule_evaluator(rule.conditions) for rule in ACTIVE_RULES
}


def _coverage(support: int) -> float:
    if TRAINING_PORTFOLIO_SIZE <= 0:
        return 0.0
    return round(support / TRAINING_PORTFOLIO_SIZE, 4)


def build_applicant_context(applicant_data: dict[str, Any]) -> dict[str, Any]:
    frame = engineer_features(pd.DataFrame([applicant_data]))
    context = frame.iloc[0].to_dict()
    context.update({k: v for k, v in applicant_data.items() if k not in context})
    return context


def rule_to_card(rule: RuleDefinition, matched: bool) -> dict[str, Any]:
    """Serialize one rule for API / dashboard consumption."""
    structured = [c.to_dict() for c in rule.conditions]
    condition_labels = [format_condition_compact(c) for c in structured]
    coverage = _coverage(rule.support)

    payload: dict[str, Any] = {
        "rule_id": rule.rule_id,
        "title": rule.title,
        "risk_band": rule.risk_band,
        "risk_band_label": f"{rule.risk_band} RISK",
        "action": rule.action,
        "conditions": structured,
        "condition_labels": condition_labels,
        "matched": matched,
        "confidence": round(rule.confidence / 100.0, 4),
        "confidence_pct": rule.confidence,
        "support": rule.support,
        "coverage": coverage,
        "lift": rule.lift,
        "precision": rule.precision,
        "recall": rule.recall,
        "why_exists": WHY_RULE_EXISTS,
    }
    payload["generated_explanation"] = generate_rule_explanation(payload)
    payload["rationale"] = generate_rule_rationale(payload)
    return payload


def evaluate_rule_cards(context: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for rule in ACTIVE_RULES:
        try:
            matched = bool(_RULE_EVALUATORS[rule.rule_id](context))
        except Exception:
            logger.exception("Rule evaluation failed for %s", rule.rule_id)
            matched = False
        cards.append(rule_to_card(rule, matched))
    return cards


def matched_rule_cards(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [card for card in evaluate_rule_cards(context) if card["matched"]]


def recommendation_from_band(risk_band: RiskBandLabel, model_decision: str) -> Recommendation:
    if risk_band == "MEDIUM":
        return "REVIEW"
    if risk_band == "HIGH" or model_decision == "REJECT":
        return "DECLINE"
    return "APPROVE"


def model_confidence(default_probability: float, threshold: float) -> int:
    distance = abs(default_probability - threshold)
    scaled = min(1.0, distance / 0.33 + 0.55)
    return int(round(min(99.0, max(55.0, scaled * 100))))


def build_decision(
    applicant_data: dict[str, Any],
    prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if prediction is None:
        prediction = predict_applicant(applicant_data)

    threshold = float(prediction["threshold"])
    default_prob = float(prediction["default_probability"])
    risk_band = prediction["risk_band"]
    model_decision = prediction["decision"]

    default_pct = round(default_prob * 100, 1)
    approval_pct = round(100.0 - default_pct, 1)
    risk_score = int(round(default_pct))

    context = build_applicant_context(applicant_data)
    all_cards = evaluate_rule_cards(context)
    fired_cards = [card for card in all_cards if card["matched"]]

    recommendation = recommendation_from_band(risk_band, model_decision)
    confidence = model_confidence(default_prob, threshold)

    if any(card["action"] == "DECLINE" for card in fired_cards) and recommendation == "APPROVE":
        recommendation = "REVIEW"
    if any(card["action"] == "APPROVE" for card in fired_cards) and recommendation == "DECLINE":
        recommendation = "REVIEW"

    logger.info(
        "Decision built | risk_score=%s band=%s recommendation=%s matched_rules=%d",
        risk_score,
        risk_band,
        recommendation,
        len(fired_cards),
    )

    return {
        "risk_score": risk_score,
        "risk_band": risk_band,
        "default_probability": default_pct,
        "approval_probability": approval_pct,
        "recommendation": recommendation,
        "confidence": confidence,
        "threshold": threshold,
        "model_decision": model_decision,
        "matched_rules": fired_cards,
        "rule_cards": all_cards,
        "active_rule_count": len(ACTIVE_RULES),
    }


def list_active_rules() -> list[dict[str, Any]]:
    return [rule_to_card(rule, matched=False) for rule in ACTIVE_RULES]


__all__ = [
    "ACTIVE_RULES",
    "RuleCondition",
    "RuleDefinition",
    "build_decision",
    "build_applicant_context",
    "build_rule_evaluator",
    "evaluate_rule_cards",
    "matched_rule_cards",
    "list_active_rules",
    "rule_to_card",
    "recommendation_from_band",
]
