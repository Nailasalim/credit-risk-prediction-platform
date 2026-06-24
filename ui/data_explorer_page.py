"""
Data Explorer — executive-grade EDA for the Home Credit training portfolio.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.portfolio_loader import resolve_portfolio_csv  # noqa: E402

PLOTLY_LAYOUT: dict[str, Any] = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "#121820",
    "font": {"color": "#c5d0e0", "size": 11},
    "margin": {"l": 48, "r": 24, "t": 52, "b": 44},
    "title": {"font": {"size": 13, "color": "#e8edf5"}},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10}},
}

TARGET_COLORS = {0: "#22c55e", 1: "#ef4444"}
TARGET_LABELS = {0: "Non-default", 1: "Default"}


@st.cache_data(show_spinner="Loading Home Credit portfolio for exploration…")
def load_data() -> pd.DataFrame:
    path = resolve_portfolio_csv()
    if path is None:
        raise FileNotFoundError(
            "application_train.csv not found. Place it at data/application_train.csv "
            "or set CREDIT_RISK_PORTFOLIO_CSV."
        )
    return pd.read_csv(path)


def _apply_filters(
    frame: pd.DataFrame,
    genders: list[str] | None,
    contract_types: list[str] | None,
) -> pd.DataFrame:
    out = frame
    if genders:
        out = out[out["CODE_GENDER"].isin(genders)]
    if contract_types:
        out = out[out["NAME_CONTRACT_TYPE"].isin(contract_types)]
    return out


def _fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def build_kpis(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    n_cols = frame.shape[1]
    default_rate = float(frame["TARGET"].mean()) * 100 if n else 0.0
    avg_income = float(frame["AMT_INCOME_TOTAL"].mean()) if n else 0.0
    return {
        "total_applications": n,
        "total_features": n_cols,
        "default_rate_pct": default_rate,
        "average_income": avg_income,
        "summary": (
            f"The portfolio contains **{n:,}** labeled credit applications across "
            f"**{n_cols}** raw features. Observed default prevalence is "
            f"**{default_rate:.1f}%**, with mean annual income of **₹{avg_income:,.0f}**. "
            "This profile reflects a heavily imbalanced binary classification problem typical "
            "of retail lending books."
        ),
    }


def build_demographics(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame.copy()
    work["age_years"] = work["DAYS_BIRTH"].abs() / 365.25
    emp = work["DAYS_EMPLOYED"].abs() / 365.25
    work["employment_years"] = emp.where(emp <= 50)

    age_fig = _fig(
        px.histogram(
            work,
            x="age_years",
            nbins=40,
            title="Age Distribution",
            labels={"age_years": "Age (years)", "count": "Applicants"},
            color_discrete_sequence=["#3b82f6"],
        )
    )

    gender_counts = (
        work["CODE_GENDER"].value_counts(dropna=False).reset_index()
    )
    gender_counts.columns = ["gender", "count"]
    gender_fig = _fig(
        px.pie(
            gender_counts,
            names="gender",
            values="count",
            hole=0.55,
            title="Gender Distribution",
            color_discrete_sequence=["#3b82f6", "#8b5cf6", "#64748b"],
        )
    )

    emp_fig = _fig(
        px.histogram(
            work.dropna(subset=["employment_years"]),
            x="employment_years",
            nbins=35,
            title="Employment Duration",
            labels={"employment_years": "Years employed", "count": "Applicants"},
            color_discrete_sequence=["#6366f1"],
        )
    )

    median_age = float(work["age_years"].median())
    insight = (
        f"Most applicants cluster around **{median_age:.0f} years** of age, representing the "
        "core credit-seeking population. Gender mix is skewed toward female applicants in this "
        "book; employment tenure shows a long tail of stable earners after filtering sentinel values."
    )
    return {"age_fig": age_fig, "gender_fig": gender_fig, "employment_fig": emp_fig, "insight": insight}


def build_financial_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame.copy()
    work["target_label"] = work["TARGET"].map(TARGET_LABELS)

    income_fig = _fig(
        px.histogram(
            work,
            x="AMT_INCOME_TOTAL",
            nbins=50,
            title="Income Distribution",
            labels={"AMT_INCOME_TOTAL": "Annual income (₹)", "count": "Applicants"},
            color_discrete_sequence=["#22c55e"],
        )
    )

    credit_fig = _fig(
        px.histogram(
            work,
            x="AMT_CREDIT",
            nbins=50,
            title="Credit Amount Distribution",
            labels={"AMT_CREDIT": "Credit amount (₹)", "count": "Applicants"},
            color_discrete_sequence=["#f59e0b"],
        )
    )

    annuity_fig = _fig(
        px.histogram(
            work,
            x="AMT_ANNUITY",
            nbins=50,
            title="Annuity Distribution",
            labels={"AMT_ANNUITY": "Annuity (₹)", "count": "Applicants"},
            color_discrete_sequence=["#a855f7"],
        )
    )

    sample = work
    if len(work) > 25_000:
        sample = work.sample(25_000, random_state=42)

    scatter_fig = _fig(
        px.scatter(
            sample,
            x="AMT_INCOME_TOTAL",
            y="AMT_CREDIT",
            color="target_label",
            opacity=0.35,
            title="Income vs Credit Amount",
            labels={
                "AMT_INCOME_TOTAL": "Annual income (₹)",
                "AMT_CREDIT": "Credit amount (₹)",
                "target_label": "Outcome",
            },
            color_discrete_map={"Non-default": "#22c55e", "Default": "#ef4444"},
        )
    )
    scatter_fig.update_traces(marker={"size": 4})

    insight = (
        "Credit amounts and annuities are right-skewed: a minority of large exposures drives tail risk. "
        "The income–credit scatter shows defaults spreading across affordability levels — "
        "supporting ratio features (income-to-credit, annuity burden) in the modeling pipeline."
    )
    return {
        "income_fig": income_fig,
        "credit_fig": credit_fig,
        "annuity_fig": annuity_fig,
        "scatter_fig": scatter_fig,
        "insight": insight,
    }


def build_risk_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame.copy()
    work["age_years"] = work["DAYS_BIRTH"].abs() / 365.25
    bins = [18, 26, 36, 46, 56, 120]
    labels = ["18–25", "26–35", "36–45", "46–55", "55+"]
    work["age_band"] = pd.cut(work["age_years"], bins=bins, labels=labels, right=False)

    def _default_by(col: str, title: str) -> go.Figure:
        agg = (
            work.groupby(col, observed=True)
            .agg(applications=("TARGET", "count"), default_rate_pct=("TARGET", lambda s: 100 * s.mean()))
            .reset_index()
            .sort_values("default_rate_pct", ascending=False)
        )
        fig = _fig(
            px.bar(
                agg,
                x=col,
                y="default_rate_pct",
                text=agg["default_rate_pct"].round(1),
                title=title,
                labels={col: col.replace("_", " ").title(), "default_rate_pct": "Default rate (%)"},
                color="default_rate_pct",
                color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            )
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        return fig, agg

    gender_fig, gender_agg = _default_by("CODE_GENDER", "Default Rate by Gender")
    contract_fig, contract_agg = _default_by("NAME_CONTRACT_TYPE", "Default Rate by Contract Type")
    age_fig, age_agg = _default_by("age_band", "Default Rate by Age Band")

    region_agg = (
        work.groupby("REGION_RATING_CLIENT", observed=True)
        .agg(applications=("TARGET", "count"), default_rate_pct=("TARGET", lambda s: 100 * s.mean()))
        .reset_index()
        .rename(columns={"REGION_RATING_CLIENT": "region_rating"})
        .sort_values("default_rate_pct", ascending=False)
    )
    region_fig = _fig(
        px.bar(
            region_agg,
            x="region_rating",
            y="default_rate_pct",
            text=region_agg["default_rate_pct"].round(1),
            title="Default Rate by Region Rating",
            labels={
                "region_rating": "Region rating",
                "default_rate_pct": "Default rate (%)",
            },
            color="default_rate_pct",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
        )
    )
    region_fig.update_traces(textposition="outside")
    region_fig.update_layout(coloraxis_showscale=False)

    highlight = contract_agg.iloc[0] if len(contract_agg) else None
    if highlight is not None:
        seg_name = str(highlight["NAME_CONTRACT_TYPE"])
        seg_rate = float(highlight["default_rate_pct"])
        seg_n = int(highlight["applications"])
        insight = (
            f"**{seg_name}** shows the highest observed default rate at **{seg_rate:.1f}%** "
            f"({seg_n:,} applications). Region rating **{int(region_agg.iloc[0]['region_rating'])}** "
            f"is the weakest geographic band at **{float(region_agg.iloc[0]['default_rate_pct']):.1f}%** default."
        )
    else:
        insight = "Insufficient filtered data to compute risk segment highlights."

    return {
        "gender_fig": gender_fig,
        "contract_fig": contract_fig,
        "age_fig": age_fig,
        "region_fig": region_fig,
        "region_agg": region_agg,
        "insight": insight,
    }


def build_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    missing_pct = (frame.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing_pct.head(15).reset_index()
    top_missing.columns = ["column", "missing_pct"]

    missing_fig = _fig(
        px.bar(
            top_missing,
            x="missing_pct",
            y="column",
            orientation="h",
            title="Top 15 Columns by Missing %",
            labels={"missing_pct": "Missing (%)", "column": "Feature"},
            color="missing_pct",
            color_continuous_scale=["#3b82f6", "#ef4444"],
        )
    )
    missing_fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})

    cols_with_missing = int((missing_pct > 0).sum())
    avg_missing = float(missing_pct.mean())
    max_missing = float(missing_pct.max()) if len(missing_pct) else 0.0
    top_col = str(missing_pct.idxmax()) if len(missing_pct) else "—"

    recommendation = (
        f"Missingness is concentrated in bureau and external-score fields (highest: **{top_col}** at "
        f"**{max_missing:.1f}%**). Median imputation on the training split is applied in the production "
        "pipeline; external sources should be monitored for data-feed stability."
    )

    return {
        "missing_fig": missing_fig,
        "cols_with_missing": cols_with_missing,
        "avg_missing_pct": avg_missing,
        "max_missing_pct": max_missing,
        "recommendation": recommendation,
    }


def inject_explorer_styles() -> None:
    st.markdown(
        """
        <style>
        .de-hero {
            margin-bottom: 1rem; padding-bottom: 0.85rem;
            border-bottom: 1px solid #243044;
        }
        .de-eyebrow {
            font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.12em; color: #3b82f6; margin: 0 0 0.35rem 0;
        }
        .de-title {
            font-size: 1.75rem; font-weight: 700; color: #f1f5f9; margin: 0;
            letter-spacing: -0.03em;
        }
        .de-sub {
            font-size: 0.88rem; color: #8b9bb4; margin: 0.4rem 0 0 0; line-height: 1.5;
        }
        .de-section {
            font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.09em; color: #4a5d78;
            margin: 1.35rem 0 0.65rem 0;
        }
        .de-kpi {
            background: linear-gradient(145deg, #161d27 0%, #131a24 100%);
            border: 1px solid #243044; border-radius: 12px;
            padding: 0.9rem 1rem; min-height: 6.25rem; height: 100%;
            box-sizing: border-box;
        }
        .de-kpi-label {
            font-size: 0.62rem; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.08em; color: #6b7c96; margin: 0 0 0.35rem 0;
        }
        .de-kpi-value {
            font-size: 1.55rem; font-weight: 700; color: #f1f5f9; margin: 0;
        }
        .de-kpi-sub { font-size: 0.68rem; color: #6b7c96; margin: 0.35rem 0 0 0; }
        .de-summary {
            font-size: 0.82rem; color: #8b9bb4; margin: 0.65rem 0 0;
            padding: 0.55rem 0.85rem; background: #121820;
            border: 1px solid #243044; border-radius: 8px; line-height: 1.5;
        }
        .de-insight {
            background: #121820; border: 1px solid #2a4060;
            border-left: 3px solid #3b82f6; border-radius: 8px;
            padding: 0.7rem 0.9rem; font-size: 0.82rem; color: #c5d0e0;
            line-height: 1.5; margin: 0.5rem 0 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, sub: str) -> str:
    return f"""
    <div class="de-kpi">
        <p class="de-kpi-label">{label}</p>
        <p class="de-kpi-value">{value}</p>
        <p class="de-kpi-sub">{sub}</p>
    </div>
    """


def _render_sidebar_filters(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    genders = sorted(frame["CODE_GENDER"].dropna().astype(str).unique().tolist())
    contracts = sorted(frame["NAME_CONTRACT_TYPE"].dropna().astype(str).unique().tolist())

    with st.sidebar:
        st.markdown("---")
        st.markdown("##### Data Explorer filters")
        sel_gender = st.multiselect(
            "Gender",
            options=genders,
            default=genders,
            key="de_filter_gender",
        )
        sel_contract = st.multiselect(
            "Contract type",
            options=contracts,
            default=contracts,
            key="de_filter_contract",
        )
    return sel_gender, sel_contract


def _plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_data_explorer_page(page_header_fn: Callable[..., None]) -> None:
    del page_header_fn
    inject_explorer_styles()

    st.markdown(
        """
        <div class="de-hero">
            <p class="de-eyebrow">Enterprise Risk Platform</p>
            <h1 class="de-title">Data Explorer</h1>
            <p class="de-sub">
                Explore applicant demographics, financial characteristics, portfolio risk patterns
                and data quality metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        raw = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    sel_gender, sel_contract = _render_sidebar_filters(raw)
    filtered = _apply_filters(raw, sel_gender or None, sel_contract or None)

    if filtered.empty:
        st.warning("No records match the current filters. Reset sidebar filters to continue.")
        return

    st.caption(f"Showing **{len(filtered):,}** of **{len(raw):,}** applications")

    # --- Section 1: Dataset Overview ---
    st.markdown('<p class="de-section">1 · Dataset overview</p>', unsafe_allow_html=True)
    kpis = build_kpis(filtered)
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        st.markdown(
            _kpi_card("Total Applications", f"{kpis['total_applications']:,}", "Labeled training rows"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _kpi_card("Total Features", f"{kpis['total_features']}", "Raw Home Credit columns"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi_card("Default Rate", f"{kpis['default_rate_pct']:.1f}%", "Observed TARGET = 1"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _kpi_card(
                "Average Income",
                f"₹{kpis['average_income']:,.0f}",
                "Mean AMT_INCOME_TOTAL",
            ),
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="de-summary">{kpis["summary"]}</div>', unsafe_allow_html=True)

    # --- Section 2: Demographics ---
    st.markdown('<p class="de-section">2 · Demographics</p>', unsafe_allow_html=True)
    demo = build_demographics(filtered)
    d1, d2, d3 = st.columns(3, gap="small")
    with d1:
        _plot(demo["age_fig"])
    with d2:
        _plot(demo["gender_fig"])
    with d3:
        _plot(demo["employment_fig"])
    with st.expander("Demographics insight", expanded=True):
        st.markdown(demo["insight"])

    # --- Section 3: Financial Analysis ---
    st.markdown('<p class="de-section">3 · Financial analysis</p>', unsafe_allow_html=True)
    fin = build_financial_analysis(filtered)
    f1, f2, f3 = st.columns(3, gap="small")
    with f1:
        _plot(fin["income_fig"])
    with f2:
        _plot(fin["credit_fig"])
    with f3:
        _plot(fin["annuity_fig"])
    _plot(fin["scatter_fig"])
    with st.expander("Financial analysis insight", expanded=True):
        st.markdown(fin["insight"])

    # --- Section 4: Risk Analysis ---
    st.markdown('<p class="de-section">4 · Risk analysis</p>', unsafe_allow_html=True)
    risk = build_risk_analysis(filtered)
    r1, r2 = st.columns(2, gap="small")
    with r1:
        _plot(risk["gender_fig"])
    with r2:
        _plot(risk["contract_fig"])
    r3, r4 = st.columns(2, gap="small")
    with r3:
        _plot(risk["age_fig"])
    with r4:
        _plot(risk["region_fig"])
    st.markdown(f'<div class="de-insight">{risk["insight"]}</div>', unsafe_allow_html=True)

    # --- Section 5: Data Quality ---
    st.markdown('<p class="de-section">5 · Data quality</p>', unsafe_allow_html=True)
    dq = build_data_quality(filtered)
    _plot(dq["missing_fig"])
    q1, q2, q3 = st.columns(3, gap="small")
    with q1:
        st.markdown(
            _kpi_card(
                "Columns With Missing Data",
                str(dq["cols_with_missing"]),
                "Features with any nulls",
            ),
            unsafe_allow_html=True,
        )
    with q2:
        st.markdown(
            _kpi_card(
                "Average Missing %",
                f"{dq['avg_missing_pct']:.1f}%",
                "Mean across all columns",
            ),
            unsafe_allow_html=True,
        )
    with q3:
        st.markdown(
            _kpi_card(
                "Highest Missing %",
                f"{dq['max_missing_pct']:.1f}%",
                "Worst single feature",
            ),
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="de-insight">{dq["recommendation"]}</div>', unsafe_allow_html=True)

    st.caption(
        "Home Credit `application_train` · filters apply to all sections · "
        "scatter plot sampled at 25k points when portfolio is large"
    )
