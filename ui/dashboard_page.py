"""
Executive Dashboard — production-grade portfolio underwriting overview (UI only).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import altair as alt
import pandas as pd
import streamlit as st

from app_navigation import request_page
from assessment_store import list_assessments
from dashboard_charts import (
    approval_gauge,
    build_chart_safe,
    configure_dashboard_chart,
    model_risk_bands_chart,
    model_risk_bands_chart_fallback,
    driver_importance_bars,
    exposure_breakdown_chart,
    risk_distribution_donut,
    simple_band_bar_chart,
    simple_decision_bar_chart,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.dashboard_summary import build_dashboard_summary  # noqa: E402
from src.ml.global_importance import load_global_importance  # noqa: E402
from src.ml.rule_explain import feature_label  # noqa: E402

RECOMMENDATION_LABEL = {"APPROVE": "Approve", "REVIEW": "Manual review", "DECLINE": "Decline"}
BAND_LABEL = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}

BAND_SCORE_MID = {"LOW": 19.5, "MEDIUM": 53.5, "HIGH": 83.5}


@st.cache_data(show_spinner="Scoring portfolio and loading dashboard metrics…")
def _load_dashboard_summary() -> dict[str, Any]:
    return build_dashboard_summary(top_features=8)


@st.cache_data(show_spinner=False)
def _load_importance_drivers() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    importance = load_global_importance()
    risk_rows = importance[importance["display_impact"] > 0].copy()
    pos_rows = importance[importance["display_impact"] < 0].copy()
    risk_rows = risk_rows.sort_values("importance", ascending=False).head(5)
    pos_rows = pos_rows.sort_values("importance", ascending=False).head(5)

    def pack(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "feature": feature_label(str(row.feature)),
                "importance": round(float(row.importance), 4),
            }
            for row in frame.itertuples()
        ]

    return pack(risk_rows), pack(pos_rows)


def inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .dash-hero {
            margin-bottom: 1.35rem; padding-bottom: 1rem;
            border-bottom: 1px solid #243044;
        }
        .dash-hero-eyebrow {
            font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.12em; color: #3b82f6; margin: 0 0 0.35rem 0;
        }
        .dash-hero-title {
            font-size: 1.85rem; font-weight: 700; color: #f1f5f9; margin: 0;
            letter-spacing: -0.03em; line-height: 1.15;
        }
        .dash-hero-sub {
            font-size: 0.92rem; color: #8b9bb4; margin: 0.45rem 0 0 0;
            max-width: 820px; line-height: 1.55;
        }
        .ex-kpi {
            background: linear-gradient(145deg, #161d27 0%, #131a24 100%);
            border: 1px solid #243044; border-radius: 12px;
            padding: 1rem 1.1rem; min-height: 7rem;
            height: 100%; box-sizing: border-box;
            display: flex; flex-direction: column; justify-content: space-between;
            transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
        }
        .ex-kpi:hover {
            border-color: #3b82f6; box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            transform: translateY(-1px);
        }
        .ex-kpi-label {
            font-size: 0.62rem; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.08em; color: #6b7c96; margin: 0 0 0.4rem 0;
        }
        .ex-kpi-value {
            font-size: 1.75rem; font-weight: 700; color: #f1f5f9; margin: 0;
            line-height: 1.05; letter-spacing: -0.02em;
        }
        .ex-kpi-sub {
            font-size: 0.68rem; color: #6b7c96; margin: 0.45rem 0 0 0; line-height: 1.4;
        }
        .ex-portfolio-summary {
            font-size: 0.8rem; color: #8b9bb4; margin: 0.75rem 0 0;
            padding: 0.55rem 0.85rem; background: #121820;
            border: 1px solid #243044; border-radius: 8px;
        }
        .ex-portfolio-summary strong { color: #e8edf5; font-weight: 600; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #121820 !important;
            border-color: #243044 !important;
            border-radius: 10px !important;
            padding: 0.35rem 0.5rem 0.15rem !important;
            margin-bottom: 0.25rem;
        }
        [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) {
            gap: 0.65rem !important;
        }
        [data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"])
        [data-testid="column"] {
            min-width: 0;
        }
        .ex-section-title {
            font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.09em; color: #4a5d78;
            margin: 1.5rem 0 0.75rem 0;
        }
        div[data-testid="stAltairChart"] {
            margin-top: 0 !important;
        }
        .insight-card {
            background: #161d27; border: 1px solid #243044; border-radius: 10px;
            padding: 0.85rem 1rem; height: 100%;
        }
        .insight-card:hover { border-color: #2f3f56; }
        .insight-label {
            font-size: 0.62rem; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.07em; color: #6b7c96; margin: 0 0 0.35rem 0;
        }
        .insight-value {
            font-size: 1.35rem; font-weight: 700; color: #e8edf5; margin: 0;
        }
        .insight-sub { font-size: 0.68rem; color: #5c6b82; margin: 0.3rem 0 0 0; }
        .badge {
            display: inline-block; font-size: 0.62rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.05em;
            padding: 0.18rem 0.45rem; border-radius: 4px;
        }
        .badge-low { background: rgba(34,197,94,0.15); color: #4ade80; }
        .badge-med { background: rgba(245,158,11,0.15); color: #fbbf24; }
        .badge-high { background: rgba(239,68,68,0.15); color: #f87171; }
        .badge-approve { background: rgba(34,197,94,0.12); color: #4ade80; }
        .badge-review { background: rgba(245,158,11,0.12); color: #fbbf24; }
        .badge-decline { background: rgba(239,68,68,0.12); color: #f87171; }
        .onboard-card {
            background: linear-gradient(135deg, #121820 0%, #161d27 100%);
            border: 1px solid #243044; border-radius: 12px;
            padding: 2rem 1.5rem; text-align: center;
        }
        .onboard-title { font-size: 1.1rem; font-weight: 700; color: #e8edf5; margin: 0 0 0.5rem 0; }
        .onboard-body { font-size: 0.85rem; color: #8b9bb4; margin: 0; line-height: 1.55; }
        .assess-table-wrap {
            border: 1px solid #243044; border-radius: 10px; overflow: hidden;
        }
        .assess-table {
            width: 100%; border-collapse: collapse; font-size: 0.78rem;
        }
        .assess-table th {
            text-align: left; padding: 0.55rem 0.75rem;
            background: #0f141c; color: #6b7c96; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.62rem;
        }
        .assess-table td {
            padding: 0.5rem 0.75rem; border-top: 1px solid #1e2a3a; color: #c5d0e0;
        }
        .assess-table tbody tr { transition: background 0.15s ease; }
        .assess-table tbody tr:hover td {
            background: rgba(59, 130, 246, 0.06);
        }
        .assess-table td:first-child { font-weight: 500; color: #e8edf5; }
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] > div > div {
            background: #121820 !important;
            border-color: #243044 !important;
            color: #e8edf5 !important;
            font-size: 0.8rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _estimated_avg_risk_score(distribution: list[dict[str, Any]]) -> float:
    total = sum(int(r["count"]) for r in distribution) or 1
    weighted = sum(BAND_SCORE_MID.get(str(r["band"]).upper(), 50) * int(r["count"]) for r in distribution)
    return round(weighted / total, 1)


def _render_platform_header() -> None:
    st.markdown(
        """
        <div class="dash-hero">
            <p class="dash-hero-eyebrow">Enterprise Risk Platform</p>
            <h1 class="dash-hero-title">Executive Dashboard</h1>
            <p class="dash-hero-sub">
                AI-powered portfolio monitoring, underwriting decisions, and risk explainability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_executive_kpis(kpis: dict[str, Any], portfolio: dict[str, Any]) -> None:
    avg_score = _estimated_avg_risk_score(portfolio.get("risk_band_distribution", []))

    specs = [
        (
            "Total Applications",
            _fmt_int(int(kpis["total_applications"])),
            "Scored applications in the monitoring book",
        ),
        (
            "Default Rate",
            _fmt_pct(float(kpis["default_rate_pct"])),
            "Observed defaults — baseline for portfolio health tracking",
        ),
        (
            "Approval Rate",
            _fmt_pct(float(kpis["approval_rate_pct"])),
            "Share approved under the current LightGBM policy threshold",
        ),
        (
            "High Risk Exposure",
            _fmt_pct(float(kpis["high_risk_pct"])),
            f"{_fmt_int(int(kpis['high_risk_count']))} accounts in the HIGH band — priority for review",
        ),
    ]

    cols = st.columns(4, gap="small")
    for col, (label, value, sub) in zip(cols, specs):
        with col:
            st.markdown(
                f"""
                <div class="ex-kpi">
                    <p class="ex-kpi-label">{label}</p>
                    <p class="ex-kpi-value">{value}</p>
                    <p class="ex-kpi-sub">{sub}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    high_pct = float(portfolio.get("high_risk_pct", 0))
    st.markdown(
        f'<p class="ex-portfolio-summary">Portfolio average modeled risk score: '
        f"<strong>{avg_score}</strong> · "
        f"<strong>{_fmt_pct(high_pct)}</strong> of volume sits in the HIGH band — "
        "the primary segment to watch for underwriting capacity and monitoring load.</p>",
        unsafe_allow_html=True,
    )


def _display_chart(chart: alt.Chart) -> None:
    st.altair_chart(configure_dashboard_chart(chart), use_container_width=True)


def _render_portfolio_analytics(portfolio: dict[str, Any]) -> None:
    st.markdown('<p class="ex-section-title">Portfolio analytics</p>', unsafe_allow_html=True)
    dist = portfolio["risk_band_distribution"]
    rec = {str(r["action"]).upper(): float(r["pct"]) for r in portfolio["recommendation_distribution"]}

    approve = rec.get("APPROVE", 0)
    decline = rec.get("DECLINE", 0)
    review = rec.get("REVIEW", 0)

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        with st.container(border=True):
            chart = build_chart_safe(
                risk_distribution_donut,
                simple_band_bar_chart,
                dist,
                primary_kwargs={"compact": True},
                fallback_kwargs={"title": "Risk Distribution", "compact": True},
            )
            _display_chart(chart)

    with c2:
        with st.container(border=True):
            chart = build_chart_safe(
                approval_gauge,
                simple_decision_bar_chart,
                approve,
                decline,
                review,
                primary_kwargs={"compact": True},
                fallback_kwargs={"title": "Approval Decision Mix", "compact": True},
            )
            _display_chart(chart)

    with c3:
        with st.container(border=True):
            chart = build_chart_safe(
                model_risk_bands_chart,
                model_risk_bands_chart_fallback,
                portfolio,
                primary_kwargs={"compact": True},
                fallback_kwargs={"compact": True},
            )
            _display_chart(chart)

    with c4:
        with st.container(border=True):
            _display_chart(exposure_breakdown_chart(dist, compact=True))

    st.caption(
        f"Batch-scored training portfolio · decision threshold {portfolio['threshold']} · "
        "Band mix (left) vs. model probability bands (centre) vs. application volume by segment (right)"
    )


def _render_underwriting_insights(
    portfolio: dict[str, Any],
    risk_drivers: list[dict[str, Any]],
    positive_drivers: list[dict[str, Any]],
) -> None:
    st.markdown('<p class="ex-section-title">Underwriting insights</p>', unsafe_allow_html=True)

    dist = portfolio["risk_band_distribution"]
    avg_score = _estimated_avg_risk_score(dist)
    high_pct = float(portfolio["high_risk_pct"])

    summary_cols = st.columns(3, gap="small")
    approve_pct = float(
        next(
            (r["pct"] for r in portfolio["recommendation_distribution"] if r["action"] == "APPROVE"),
            0.0,
        )
    )
    summaries = [
        (
            "Average Risk Score",
            f"{avg_score}",
            "Executive index summarizing modeled risk across the full scored book",
        ),
        (
            "High-Risk Population",
            _fmt_pct(high_pct),
            "Applicants in the HIGH band — focus for escalation, policy tuning, and watchlists",
        ),
        (
            "Portfolio Risk Summary",
            f"{_fmt_int(int(portfolio['high_risk_count']))} high-risk accounts",
            f"{_fmt_pct(high_pct)} of volume in HIGH band · {_fmt_pct(approve_pct)} approved under current policy",
        ),
    ]
    for col, (label, value, sub) in zip(summary_cols, summaries):
        with col:
            st.markdown(
                f"""
                <div class="insight-card">
                    <p class="insight-label">{label}</p>
                    <p class="insight-value">{value}</p>
                    <p class="insight-sub">{sub}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="small")

    def _safe_driver_chart(drivers: list[dict], title: str, color: str) -> None:
        with st.container(border=True):
            try:
                chart = driver_importance_bars(drivers, title=title, color=color)
                st.altair_chart(configure_dashboard_chart(chart), use_container_width=True)
            except Exception:
                if drivers:
                    fallback = pd.DataFrame(drivers)
                    st.bar_chart(fallback.set_index("feature")[["importance"]], height=200)
                else:
                    st.caption("Driver chart unavailable.")

    with d1:
        _safe_driver_chart(risk_drivers, "Top Risk Drivers", "#f87171")
    with d2:
        _safe_driver_chart(positive_drivers, "Top Positive Drivers", "#4ade80")

    st.caption("Drivers from global mean |SHAP| (portfolio validation sample).")


def _badge_html(kind: str, text: str) -> str:
    css = {
        "LOW": "badge-low",
        "MEDIUM": "badge-med",
        "HIGH": "badge-high",
        "APPROVE": "badge-approve",
        "REVIEW": "badge-review",
        "DECLINE": "badge-decline",
    }.get(kind.upper(), "badge-med")
    return f'<span class="badge {css}">{text}</span>'


def _render_recent_assessments() -> None:
    st.markdown('<p class="ex-section-title">Recent assessments</p>', unsafe_allow_html=True)
    assessments = list_assessments()

    if not assessments:
        st.markdown(
            """
            <div class="onboard-card">
                <p class="onboard-title">No live assessments yet</p>
                <p class="onboard-body">
                    Run an applicant through <strong>Risk Prediction</strong> to populate this feed
                    with scored decisions, risk bands, and explainability-ready records.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Run Credit Assessment", type="secondary", use_container_width=False):
            request_page("risk_prediction")
            st.rerun()
        return

    c1, c2, c3 = st.columns((2, 1, 1), gap="small")
    with c1:
        search = st.text_input(
            "Search assessments",
            placeholder="Applicant name or ID…",
            label_visibility="collapsed",
        )
    with c2:
        sort_col = st.selectbox(
            "Sort by",
            ["Assessed (UTC)", "Risk score", "Applicant", "Default %"],
            label_visibility="collapsed",
        )
    with c3:
        sort_dir = st.selectbox("Order", ["Descending", "Ascending"], label_visibility="collapsed")

    rows = []
    for item in assessments:
        rec = str(item.get("recommendation", "")).upper()
        band = str(item.get("risk_band", "MEDIUM")).upper()
        score = int(item.get("risk_score", 0))
        rows.append(
            {
                "Applicant": item.get("applicant_name", "—"),
                "ID": item.get("applicant_id", "—"),
                "Risk score": score,
                "Band": band,
                "Default %": float(item.get("default_probability_pct", item.get("default_probability", 0))),
                "Recommendation": RECOMMENDATION_LABEL.get(rec, rec.title()),
                "RecKey": rec,
                "Assessed (UTC)": item.get("timestamp") or item.get("recorded_at", ""),
            }
        )

    frame = pd.DataFrame(rows)
    if search.strip():
        q = search.strip().lower()
        frame = frame[
            frame["Applicant"].str.lower().str.contains(q, na=False)
            | frame["ID"].str.lower().str.contains(q, na=False)
        ]

    ascending = sort_dir == "Ascending"
    sort_map = {
        "Assessed (UTC)": "Assessed (UTC)",
        "Risk score": "Risk score",
        "Applicant": "Applicant",
        "Default %": "Default %",
    }
    frame = frame.sort_values(sort_map[sort_col], ascending=ascending)

    if frame.empty:
        st.info("No assessments match your search.")
        return

    header = (
        "<tr><th>Applicant</th><th>ID</th><th>Risk</th><th>Band</th>"
        "<th>Default</th><th>Recommendation</th><th>Assessed (UTC)</th></tr>"
    )
    body_rows = []
    for _, row in frame.iterrows():
        body_rows.append(
            "<tr>"
            f"<td>{row['Applicant']}</td>"
            f"<td style='color:#6b7c96'>{row['ID']}</td>"
            f"<td><strong>{int(row['Risk score'])}</strong></td>"
            f"<td>{_badge_html(row['Band'], BAND_LABEL.get(row['Band'], row['Band']))}</td>"
            f"<td>{float(row['Default %']):.1f}%</td>"
            f"<td>{_badge_html(row['RecKey'], row['Recommendation'])}</td>"
            f"<td style='color:#6b7c96;font-size:0.72rem'>{row['Assessed (UTC)']}</td>"
            "</tr>"
        )

    st.markdown(
        f'<div class="assess-table-wrap"><table class="assess-table">{header}{"".join(body_rows)}</table></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Session assessments from Risk Prediction · {len(frame)} shown")


def render_executive_dashboard(page_header_fn: Callable[..., None]) -> None:
    del page_header_fn  # Custom platform header replaces generic page header
    inject_dashboard_styles()
    try:
        alt.themes.enable("default")
    except Exception:
        pass

    _render_platform_header()

    try:
        summary = _load_dashboard_summary()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Unable to load dashboard: {exc}")
        return

    portfolio = summary["portfolio"]
    risk_drv, pos_drv = _load_importance_drivers()

    _render_executive_kpis(summary["executive_kpis"], portfolio)
    _render_portfolio_analytics(portfolio)
    _render_underwriting_insights(portfolio, risk_drv, pos_drv)
    _render_recent_assessments()

    runtime = summary.get("scoring_runtime_seconds")
    if runtime is not None:
        st.caption(
            f"Portfolio metrics cached · scored in {runtime}s · "
            "models/portfolio_scoring_snapshot.json"
        )
