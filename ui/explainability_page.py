"""
Explainability page — compact underwriting explanation view (UI only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app_navigation import EXPLAIN_MODE_CUSTOM, EXPLAIN_MODE_LATEST
from assessment_store import get_latest_assessment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import preprocess_applicant  # noqa: E402
from src.ml.rule_explain import feature_label as _feature_label  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
MODEL_PATH = MODELS_DIR / "model.pkl"

FEATURE_LABELS: dict[str, str] = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Credit amount",
    "AMT_ANNUITY": "Monthly EMI",
    "AMT_GOODS_PRICE": "Goods price",
    "DAYS_BIRTH": "Applicant age",
    "DAYS_EMPLOYED": "Employment history",
    "REGION_RATING_CLIENT": "Region rating",
    "REGION_RATING_CLIENT_W_CITY": "Region & city rating",
    "INCOME_CREDIT_RATIO": "Income-to-credit ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-income ratio",
    "CREDIT_GOODS_RATIO": "Loan-to-Goods Ratio",
    "DAYS_LAST_PHONE_CHANGE": "Phone stability",
    "DAYS_ID_PUBLISH": "ID tenure",
    "REG_CITY_NOT_WORK_CITY": "Work vs registered address",
    "REG_CITY_NOT_LIVE_CITY": "Residence vs registered address",
    "FLAG_EMP_PHONE": "Employer phone",
    "FLAG_DOCUMENT_3": "ID document",
    "OWN_CAR_AGE": "Vehicle age",
}

BAND_COLORS = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
RECOMMENDATION_LABEL = {"APPROVE": "Approve", "REVIEW": "Manual review", "DECLINE": "Decline"}


# ---------------------------------------------------------------------------
# Data (unchanged logic)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model() -> Any:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_names() -> list[str]:
    with FEATURE_NAMES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def compute_contributions(applicant_payload: dict[str, Any]) -> pd.DataFrame:
    model = load_model()
    features = preprocess_applicant(applicant_payload)
    raw = model.predict(features, pred_contrib=True)[0]
    names = load_feature_names()
    values = raw[: len(names)]
    frame = pd.DataFrame({"feature": names, "impact": values})
    frame["abs_impact"] = frame["impact"].abs()
    return frame.sort_values("abs_impact", ascending=False).reset_index(drop=True)


def _driver_label(feature: str, impact: float, payload: dict[str, Any]) -> str:
    val = payload.get(feature)
    if feature == "EXT_SOURCE_2" and val is not None and float(val) < 0.5:
        return "Low External Credit Score 2"
    if feature == "EXT_SOURCE_3" and val is not None and float(val) < 0.5:
        return "Low External Credit Score 3"
    if feature == "EXT_SOURCE_1" and val is not None and float(val) < 0.5:
        return "Low External Credit Score 1"
    if feature == "DAYS_EMPLOYED" and val is not None and abs(int(val)) / 365.25 < 2:
        return "Short employment history"
    if feature == "ANNUITY_INCOME_RATIO" and val is not None and float(val) > 0.30:
        return "High annuity burden"
    if feature == "INCOME_CREDIT_RATIO" and val is not None and float(val) < 0.2:
        return "High credit burden"
    if feature.startswith("EXT_SOURCE") and val is not None and float(val) >= 0.65 and impact < 0:
        return "Strong external credit history"
    return FEATURE_LABELS.get(feature, _feature_label(feature))


def split_drivers(contributions: pd.DataFrame, payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    risk_titles = [
        _driver_label(str(row.feature), float(row.impact), payload)
        for row in contributions[contributions["impact"] > 0].head(3).itertuples()
    ]
    positive_titles = [
        _driver_label(str(row.feature), float(row.impact), payload)
        for row in contributions[contributions["impact"] < 0].head(3).itertuples()
    ]
    return risk_titles, positive_titles


def build_concise_summary(
    risk_band: str,
    risk_drivers: list[str],
    positive_factors: list[str],
) -> str:
    """Two to three sentences for the AI Summary card."""
    band = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}.get(risk_band.upper(), risk_band.lower())
    if not risk_drivers:
        return (
            f"This applicant is assessed as {band} risk with no dominant adverse drivers "
            "in the local SHAP profile."
        )
    lead = risk_drivers[0].lower()
    text = (
        f"This applicant is assessed as {band} risk, primarily influenced by {lead}"
    )
    if len(risk_drivers) > 1:
        text += f" and {risk_drivers[1].lower()}"
    text += "."
    if positive_factors:
        text += f" {positive_factors[0]} partially offsets default risk."
    return text


def _has_latest_assessment() -> bool:
    return (
        st.session_state.get("last_applicant_payload") is not None
        and st.session_state.get("risk_result") is not None
    )


def _init_explainability_mode() -> None:
    if st.session_state.get("explainability_open_latest"):
        st.session_state.explainability_mode = EXPLAIN_MODE_LATEST
        st.session_state.pop("explainability_open_latest", None)
    elif "explainability_mode" not in st.session_state:
        st.session_state.explainability_mode = (
            EXPLAIN_MODE_LATEST if _has_latest_assessment() else EXPLAIN_MODE_CUSTOM
        )


def _badge_class_risk(band: str) -> str:
    return {"LOW": "xai-badge-low", "MEDIUM": "xai-badge-med", "HIGH": "xai-badge-high"}.get(
        band.upper(), "xai-badge-med"
    )


def _badge_class_rec(rec: str) -> str:
    return {
        "APPROVE": "xai-badge-approve",
        "REVIEW": "xai-badge-review",
        "DECLINE": "xai-badge-decline",
    }.get(rec.upper(), "xai-badge-review")


def _default_probability_pct(decision: dict[str, Any]) -> str:
    raw = decision.get("default_probability", decision.get("default_probability_pct", 0))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return "—"
    if value <= 1.0:
        value *= 100.0
    return f"{value:.1f}%"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def inject_explainability_styles() -> None:
    st.markdown(
        """
        <style>
        .xai-hero {
            margin-bottom: 0.75rem; padding-bottom: 0.65rem;
            border-bottom: 1px solid #243044;
        }
        .xai-hero-eyebrow {
            font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.12em; color: #3b82f6; margin: 0 0 0.3rem 0;
        }
        .xai-hero-title {
            font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin: 0;
            letter-spacing: -0.03em;
        }
        .xai-hero-sub {
            font-size: 0.85rem; color: #8b9bb4; margin: 0.35rem 0 0 0; line-height: 1.45;
        }
        .xai-applicant-line {
            font-size: 0.78rem; color: #94a3b8; margin: 0.5rem 0 0.15rem 0;
        }
        .xai-card {
            background: linear-gradient(145deg, #161d27 0%, #131a24 100%);
            border: 1px solid #243044; border-radius: 10px;
            padding: 0.75rem 0.9rem; margin-bottom: 0.5rem;
        }
        .xai-card-title {
            font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.07em; color: #6b7c96; margin: 0;
        }
        .xai-card-value {
            font-size: 1.35rem; font-weight: 700; color: #f1f5f9;
            margin: 0.3rem 0 0 0; line-height: 1.1;
        }
        .xai-card-sub {
            font-size: 0.68rem; color: #6b7c96; margin: 0.35rem 0 0 0;
        }
        .xai-badge-risk {
            display: inline-block; font-size: 0.6rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.05em;
            padding: 0.15rem 0.45rem; border-radius: 4px; margin-top: 0.35rem;
        }
        .xai-badge-low { background: rgba(34,197,94,0.15); color: #4ade80; }
        .xai-badge-med { background: rgba(245,158,11,0.15); color: #fbbf24; }
        .xai-badge-high { background: rgba(239,68,68,0.15); color: #f87171; }
        .xai-badge-approve { background: rgba(34,197,94,0.12); color: #4ade80; }
        .xai-badge-review { background: rgba(245,158,11,0.12); color: #fbbf24; }
        .xai-badge-decline { background: rgba(239,68,68,0.12); color: #f87171; }
        .xai-summary-body {
            font-size: 0.82rem; color: #c5d0e0; line-height: 1.5; margin: 0.35rem 0 0 0;
        }
        .xai-driver-list { margin: 0.4rem 0 0 0; padding: 0; list-style: none; }
        .xai-driver-list li {
            font-size: 0.78rem; color: #d1dae8; padding: 0.28rem 0;
            border-bottom: 1px solid #1e2a3a;
        }
        .xai-driver-list li:last-child { border-bottom: none; }
        .xai-driver-list.risk li { border-left: 2px solid #ef4444; padding-left: 0.45rem; }
        .xai-driver-list.pos li { border-left: 2px solid #22c55e; padding-left: 0.45rem; }
        .xai-chart-panel {
            background: #121820; border: 1px solid #243044; border-radius: 10px;
            padding: 0.5rem 0.65rem 0.25rem; margin-bottom: 0.35rem;
        }
        .xai-onboard {
            background: linear-gradient(135deg, #121820 0%, #161d27 100%);
            border: 1px solid #243044; border-radius: 10px;
            padding: 1.5rem 1rem; text-align: center;
        }
        .xai-onboard h3 { color: #e8edf5; font-size: 1rem; margin: 0 0 0.4rem 0; }
        .xai-onboard p { color: #8b9bb4; font-size: 0.82rem; margin: 0; line-height: 1.5; }
        div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 0.3rem; }
        div[data-testid="stRadio"] label {
            background: #121820; border: 1px solid #243044; border-radius: 8px;
            padding: 0.3rem 0.55rem !important; font-size: 0.76rem;
        }
        div[data-testid="stRadio"] label[data-checked="true"] {
            border-color: #3b82f6; background: rgba(59,130,246,0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_shap_contribution_chart(contributions: pd.DataFrame) -> None:
    frame = contributions.head(10).copy()
    frame["label"] = frame["feature"].map(lambda f: _feature_label(str(f)))
    frame["direction"] = np.where(frame["impact"] > 0, "Increases risk", "Decreases risk")

    st.markdown('<p class="xai-card-title" style="margin:0 0 0.35rem 0.15rem;">SHAP Feature Contribution</p>', unsafe_allow_html=True)
    try:
        chart = (
            alt.Chart(frame)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                y=alt.Y(
                    "label:N",
                    sort=alt.EncodingSortField(field="abs_impact", order="descending"),
                    title=None,
                    axis=alt.Axis(labelLimit=240, labelFontSize=11),
                ),
                x=alt.X("impact:Q", title="Contribution to default risk"),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["Increases risk", "Decreases risk"],
                        range=["#f87171", "#4ade80"],
                    ),
                    legend=alt.Legend(title=None, orient="top", labelFontSize=11),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Feature"),
                    alt.Tooltip("impact:Q", title="SHAP", format=".3f"),
                ],
            )
            .properties(
                height=300,
                padding={"left": 8, "right": 12, "top": 4, "bottom": 8},
            )
        )
        st.altair_chart(chart.configure(background="transparent"), use_container_width=True)
    except Exception:
        st.bar_chart(frame.set_index("label")[["impact"]], height=260)


def render_metrics_row(decision: dict[str, Any]) -> None:
    band = str(decision.get("risk_band", "MEDIUM")).upper()
    color = BAND_COLORS.get(band, "#f59e0b")
    score = decision.get("risk_score", "—")
    default_pct = _default_probability_pct(decision)
    rec_key = str(decision.get("recommendation", decision.get("model_decision", ""))).upper()
    rec_label = RECOMMENDATION_LABEL.get(rec_key, rec_key.title() if rec_key else "—")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="xai-card">
                <p class="xai-card-title">Risk Score</p>
                <p class="xai-card-value" style="color:{color};">{score}</p>
                <span class="xai-badge-risk {_badge_class_risk(band)}">{band} Risk</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="xai-card">
                <p class="xai-card-title">Recommendation</p>
                <p class="xai-card-value" style="font-size:1.1rem;">{rec_label}</p>
                <span class="xai-badge-risk {_badge_class_rec(rec_key)}">{rec_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="xai-card">
                <p class="xai-card-title">Default Probability</p>
                <p class="xai-card-value">{default_pct}</p>
                <p class="xai-card-sub">Model-estimated default likelihood</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_driver_lists(risk_drivers: list[str], positive_factors: list[str]) -> None:
    risk_items = "".join(f"<li>{title}</li>" for title in risk_drivers) or "<li>None identified</li>"
    pos_items = "".join(f"<li>{title}</li>" for title in positive_factors) or "<li>None identified</li>"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="xai-card">
                <p class="xai-card-title">Risk Drivers</p>
                <ul class="xai-driver-list risk">{risk_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="xai-card">
                <p class="xai-card-title">Positive Factors</p>
                <ul class="xai-driver-list pos">{pos_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ai_summary(text: str) -> None:
    st.markdown(
        f"""
        <div class="xai-card">
            <p class="xai-card-title">AI Summary</p>
            <p class="xai-summary-body">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_explain_state(*, latest_mode: bool) -> None:
    st.markdown(
        """
        <div class="xai-onboard">
            <h3>No applicant selected</h3>
            <p>Run a credit assessment or explain a custom applicant profile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if latest_mode:
        from app_navigation import navigate_to_explainability_latest

        if st.button("Open latest assessment", type="secondary"):
            navigate_to_explainability_latest()
            st.rerun()
    elif st.button("Run Credit Assessment", type="secondary"):
        from app_navigation import request_page

        request_page("risk_prediction")
        st.rerun()


def render_mode_selector() -> str:
    _init_explainability_mode()
    has_latest = _has_latest_assessment()
    options = [EXPLAIN_MODE_LATEST, EXPLAIN_MODE_CUSTOM] if has_latest else [EXPLAIN_MODE_CUSTOM]
    if st.session_state.explainability_mode not in options:
        st.session_state.explainability_mode = options[0]
    return st.radio(
        "Explanation source",
        options=options,
        horizontal=True,
        label_visibility="collapsed",
        key="explainability_mode",
    )


def _resolve_applicant_context(
    mode: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if mode == EXPLAIN_MODE_LATEST:
        if not _has_latest_assessment():
            return None, None, None
        latest = get_latest_assessment()
        return (
            dict(st.session_state["last_applicant_payload"]),
            dict(st.session_state["risk_result"]),
            latest,
        )
    payload = st.session_state.get("custom_explain_payload")
    if payload:
        return dict(payload), None, None
    return None, None, None


def render_custom_applicant_form() -> None:
    from risk_prediction_page import DEFAULT_APPLICANT, _days_employed_from_years, _years_from_days_employed

    d = DEFAULT_APPLICANT
    with st.form("explain_custom_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            ext2 = st.number_input("External credit score 2", 0.0, 1.0, float(d["EXT_SOURCE_2"]), 0.01)
            ext3 = st.number_input("External credit score 3", 0.0, 1.0, float(d["EXT_SOURCE_3"]), 0.01)
            income = st.number_input("Annual income (₹)", 0.0, float(d["AMT_INCOME_TOTAL"]), step=10000.0)
        with c2:
            ext1 = st.number_input("External credit score 1", 0.0, 1.0, float(d["EXT_SOURCE_1"]), 0.01)
            credit = st.number_input("Credit amount (₹)", 0.0, float(d["AMT_CREDIT"]), step=10000.0)
            years = st.number_input(
                "Employment years",
                0.0,
                50.0,
                _years_from_days_employed(int(d["DAYS_EMPLOYED"])),
                0.5,
            )
        submitted = st.form_submit_button("Generate explanation", type="primary", use_container_width=True)

    if submitted:
        st.session_state.custom_explain_payload = {
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": float(d["AMT_ANNUITY"]),
            "AMT_GOODS_PRICE": float(d["AMT_GOODS_PRICE"]),
            "DAYS_BIRTH": int(d["DAYS_BIRTH"]),
            "DAYS_EMPLOYED": _days_employed_from_years(years),
            "REGION_RATING_CLIENT": int(d["REGION_RATING_CLIENT"]),
            "REGION_RATING_CLIENT_W_CITY": int(d["REGION_RATING_CLIENT_W_CITY"]),
            "DAYS_LAST_PHONE_CHANGE": int(d["DAYS_LAST_PHONE_CHANGE"]),
            "DAYS_ID_PUBLISH": int(d["DAYS_ID_PUBLISH"]),
            "REG_CITY_NOT_WORK_CITY": int(d["REG_CITY_NOT_WORK_CITY"]),
            "REG_CITY_NOT_LIVE_CITY": int(d["REG_CITY_NOT_LIVE_CITY"]),
            "FLAG_EMP_PHONE": int(d["FLAG_EMP_PHONE"]),
            "FLAG_DOCUMENT_3": int(d["FLAG_DOCUMENT_3"]),
            "OWN_CAR_AGE": float(d["OWN_CAR_AGE"]),
        }
        st.rerun()


def _estimate_decision_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    features = preprocess_applicant(payload)
    prob = float(model.predict_proba(features)[0][1])
    pct = round(prob * 100, 1)
    if prob >= 0.67:
        band = "HIGH"
    elif prob >= 0.40:
        band = "MEDIUM"
    else:
        band = "LOW"
    return {
        "risk_band": band,
        "default_probability": pct,
        "risk_score": int(round(pct)),
        "recommendation": "DECLINE" if band == "HIGH" else "REVIEW" if band == "MEDIUM" else "APPROVE",
    }


def render_explainability_content() -> None:
    mode = render_mode_selector()

    if mode == EXPLAIN_MODE_LATEST:
        st.session_state.pop("custom_explain_payload", None)
    else:
        render_custom_applicant_form()

    payload, decision, latest_meta = _resolve_applicant_context(mode)
    if not payload:
        render_empty_explain_state(latest_mode=(mode == EXPLAIN_MODE_LATEST))
        return

    try:
        contributions = compute_contributions(payload)
    except Exception as exc:
        st.error(f"Explanation unavailable: {exc}")
        return

    if decision is None:
        decision = _estimate_decision_from_payload(payload)

    if mode == EXPLAIN_MODE_LATEST and latest_meta:
        name = latest_meta.get("applicant_name", "—")
        app_id = latest_meta.get("applicant_id", "—")
        st.markdown(
            f'<p class="xai-applicant-line">Applicant: <strong>{name}</strong> · ID {app_id}</p>',
            unsafe_allow_html=True,
        )

    render_metrics_row(decision)

    risk_titles, positive_titles = split_drivers(contributions, payload)
    summary = build_concise_summary(
        str(decision.get("risk_band", "MEDIUM")),
        risk_titles,
        positive_titles,
    )
    render_ai_summary(summary)
    render_driver_lists(risk_titles, positive_titles)

    st.markdown('<div class="xai-chart-panel">', unsafe_allow_html=True)
    render_shap_contribution_chart(contributions)
    st.markdown("</div>", unsafe_allow_html=True)


def render_explainability_page(page_header_fn: Callable[..., None]) -> None:
    del page_header_fn
    inject_explainability_styles()

    st.markdown(
        """
        <div class="xai-hero">
            <p class="xai-hero-eyebrow">Enterprise Risk Platform</p>
            <h1 class="xai-hero-title">Explainability</h1>
            <p class="xai-hero-sub">Local SHAP drivers and underwriting context for the selected applicant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_explainability_content()
