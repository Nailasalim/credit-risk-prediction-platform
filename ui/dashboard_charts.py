"""
Altair chart builders for the executive dashboard (presentation only).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import altair as alt
import pandas as pd

_COMPACT_HEIGHT = 252
_STANDARD_HEIGHT = 280

_BAND_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#f59e0b",
    "HIGH": "#ef4444",
}

_REC_COLORS = {
    "APPROVE": "#22c55e",
    "REVIEW": "#f59e0b",
    "DECLINE": "#ef4444",
}


def _chart_padding(*, compact: bool = False) -> dict[str, int]:
    if compact:
        return {"left": 4, "right": 4, "top": 4, "bottom": 4}
    return {"left": 8, "right": 8, "top": 8, "bottom": 8}


def _chart_title(title: str, *, compact: bool = False) -> alt.TitleParams:
    """Readable chart titles that avoid truncation in narrow columns."""
    return alt.TitleParams(
        text=title,
        fontSize=11 if compact else 12,
        fontWeight=600,
        color="#8b9bb4",
        anchor="start",
        offset=4,
    )


def _single_chart_props(height: int, title: str, *, compact: bool = False) -> dict[str, Any]:
    """Padding on standalone charts only (not on LayerChart children)."""
    return {
        "height": height,
        "padding": _chart_padding(compact=compact),
        "title": _chart_title(title, compact=compact),
    }


def _chart_height(*, compact: bool) -> int:
    return _COMPACT_HEIGHT if compact else _STANDARD_HEIGHT


def _count_axis(**kwargs: Any) -> alt.Axis:
    """Integer counts without scientific notation."""
    return alt.Axis(format=",", labelFontSize=10, labelPadding=4, **kwargs)


def _pct_axis(**kwargs: Any) -> alt.Axis:
    return alt.Axis(format=".0f", labelFontSize=10, labelPadding=4, **kwargs)


def _category_axis(*, title: str | None = None, label_angle: int = 0) -> alt.Axis:
    return alt.Axis(
        title=title,
        labelAngle=label_angle,
        labelFontSize=10,
        labelLimit=160,
        labelPadding=6,
    )


def configure_dashboard_chart(chart: alt.Chart) -> alt.Chart:
    """Theme wrapper safe across Altair versions."""
    return chart.configure(background="transparent")


def simple_band_bar_chart(
    distribution: list[dict[str, Any]],
    *,
    title: str,
    value_field: str = "count",
    compact: bool = False,
) -> alt.Chart:
    """Stable bar-chart fallback when arc charts fail."""
    order = ["LOW", "MEDIUM", "HIGH"]
    frame = pd.DataFrame(distribution)
    frame["band"] = pd.Categorical(frame["band"].str.upper(), categories=order, ordered=True)
    frame = frame.sort_values("band")
    frame["label"] = frame["band"].astype(str).str.title()

    color_scale = alt.Scale(
        domain=frame["label"].tolist(),
        range=[_BAND_COLORS[b] for b in order],
    )

    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("label:N", sort=None, title="Risk band", axis=_category_axis()),
            y=alt.Y(f"{value_field}:Q", title="Applications", axis=_count_axis()),
            color=alt.Color("label:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("label:N", title="Band"),
                alt.Tooltip(f"{value_field}:Q", title="Value", format=","),
            ],
        )
        .properties(**_single_chart_props(_chart_height(compact=compact), title, compact=compact))
    )


def simple_decision_bar_chart(
    approval_pct: float,
    decline_pct: float,
    review_pct: float,
    *,
    title: str,
    compact: bool = False,
) -> alt.Chart:
    frame = pd.DataFrame(
        [
            {"segment": "Approve", "pct": approval_pct},
            {"segment": "Review", "pct": review_pct},
            {"segment": "Decline", "pct": decline_pct},
        ]
    )
    color_scale = alt.Scale(
        domain=["Approve", "Review", "Decline"],
        range=[_REC_COLORS["APPROVE"], _REC_COLORS["REVIEW"], _REC_COLORS["DECLINE"]],
    )
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("segment:N", sort=None, title="Decision", axis=_category_axis()),
            y=alt.Y("pct:Q", title="Portfolio share (%)", axis=_pct_axis()),
            color=alt.Color("segment:N", scale=color_scale, legend=None),
            tooltip=[alt.Tooltip("segment:N"), alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(**_single_chart_props(_chart_height(compact=compact), title, compact=compact))
    )


def risk_distribution_donut(
    distribution: list[dict[str, Any]],
    *,
    compact: bool = False,
) -> alt.Chart:
    """Single-layer donut (no LayerChart — avoids Altair padding error)."""
    order = ["LOW", "MEDIUM", "HIGH"]
    frame = pd.DataFrame(distribution)
    frame["band"] = pd.Categorical(frame["band"], categories=order, ordered=True)
    frame = frame.sort_values("band")
    frame["label"] = frame["band"].str.title()

    color_scale = alt.Scale(
        domain=frame["label"].tolist(),
        range=[_BAND_COLORS[b] for b in order],
    )

    legend = None if compact else alt.Legend(
        title="Underwriting band",
        orient="bottom",
        labelFontSize=10,
        symbolSize=80,
        columns=3,
    )
    inner, outer = (48, 88) if compact else (58, 108)

    return (
        alt.Chart(frame)
        .mark_arc(innerRadius=inner, outerRadius=outer, padAngle=0.02)
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color("label:N", scale=color_scale, legend=legend),
            tooltip=[
                alt.Tooltip("label:N", title="Band"),
                alt.Tooltip("count:Q", title="Applications", format=","),
                alt.Tooltip("pct:Q", title="Share", format=".1f"),
            ],
        )
        .properties(
            **_single_chart_props(_chart_height(compact=compact), "Risk Distribution", compact=compact)
        )
    )


def approval_gauge(
    approval_pct: float,
    decline_pct: float,
    review_pct: float,
    *,
    compact: bool = False,
) -> alt.Chart:
    """Single-layer decision arc (no LayerChart)."""
    frame = pd.DataFrame(
        [
            {"segment": "Approve", "pct": approval_pct, "order": 0},
            {"segment": "Review", "pct": review_pct, "order": 1},
            {"segment": "Decline", "pct": decline_pct, "order": 2},
        ]
    )
    color_scale = alt.Scale(
        domain=["Approve", "Review", "Decline"],
        range=[_REC_COLORS["APPROVE"], _REC_COLORS["REVIEW"], _REC_COLORS["DECLINE"]],
    )

    legend = None if compact else alt.Legend(title="Decision", orient="bottom", labelFontSize=10)
    inner, outer = (58, 92) if compact else (72, 110)

    return (
        alt.Chart(frame)
        .mark_arc(innerRadius=inner, outerRadius=outer, padAngle=0.03)
        .encode(
            theta=alt.Theta("pct:Q", stack=True),
            color=alt.Color("segment:N", scale=color_scale, legend=legend),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("segment:N", title="Outcome"),
                alt.Tooltip("pct:Q", title="Share (%)", format=".1f"),
            ],
        )
        .properties(
            **_single_chart_props(
                _chart_height(compact=compact), "Approval Decision Mix", compact=compact
            )
        )
    )


def _resolve_probability_buckets(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    buckets = portfolio.get("probability_bucket_distribution")
    if not buckets:
        raise ValueError("probability_bucket_distribution missing from portfolio payload")
    return list(buckets)


def model_risk_bands_chart(
    portfolio: dict[str, Any],
    *,
    compact: bool = False,
) -> alt.Chart:
    """Predicted default probability buckets across the scored portfolio."""
    bucket_order = ["0–20%", "20–40%", "40–67%", "67%+"]
    frame = pd.DataFrame(_resolve_probability_buckets(portfolio))
    total = int(frame["count"].sum()) or 1
    frame["pct"] = (frame["count"] / total * 100).round(1)
    frame["bucket"] = pd.Categorical(frame["bucket"], categories=bucket_order, ordered=True)
    frame = frame.sort_values("bucket")

    bucket_colors = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444"]
    color_scale = alt.Scale(domain=bucket_order, range=bucket_colors)

    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "bucket:N",
                sort=bucket_order,
                title="Predicted default probability",
                axis=_category_axis(label_angle=0),
            ),
            y=alt.Y("count:Q", title="Applications", axis=_count_axis()),
            color=alt.Color("bucket:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("bucket:N", title="Probability band"),
                alt.Tooltip("count:Q", title="Applications", format=","),
                alt.Tooltip("pct:Q", title="Portfolio share (%)", format=".1f"),
            ],
        )
        .properties(
            **_single_chart_props(
                _chart_height(compact=compact),
                "Predicted Default Probability Bands",
                compact=compact,
            )
        )
    )


def default_probability_histogram(
    portfolio: dict[str, Any],
    *,
    compact: bool = False,
) -> alt.Chart:
    """Backward-compatible alias for the model probability bands chart."""
    return model_risk_bands_chart(portfolio, compact=compact)


def model_risk_bands_chart_fallback(
    portfolio: dict[str, Any],
    *,
    compact: bool = False,
) -> alt.Chart:
    """Fallback when probability buckets are unavailable (legacy snapshot)."""
    return simple_band_bar_chart(
        portfolio.get("risk_band_distribution", []),
        title="Predicted Default Probability Bands",
        compact=compact,
    )


def exposure_breakdown_chart(
    distribution: list[dict[str, Any]],
    *,
    compact: bool = False,
) -> alt.Chart:
    """Application volume by risk segment (concentration view, not band mix)."""
    frame = pd.DataFrame(distribution).copy()
    frame["band_key"] = frame["band"].str.upper()
    frame["band_label"] = frame["band_key"].map(
        {
            "LOW": "Low segment",
            "MEDIUM": "Medium segment",
            "HIGH": "High segment",
        }
    )
    frame["band_label"] = pd.Categorical(
        frame["band_label"],
        categories=["High segment", "Medium segment", "Low segment"],
        ordered=True,
    )
    frame = frame.sort_values("band_label", ascending=True)

    color_scale = alt.Scale(
        domain=["High segment", "Medium segment", "Low segment"],
        range=[_BAND_COLORS["HIGH"], _BAND_COLORS["MEDIUM"], _BAND_COLORS["LOW"]],
    )

    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y(
                "band_label:N",
                sort=["High segment", "Medium segment", "Low segment"],
                title="Risk segment",
                axis=_category_axis(),
            ),
            x=alt.X("count:Q", title="Application volume", axis=_count_axis()),
            color=alt.Color("band_label:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("band_label:N", title="Segment"),
                alt.Tooltip("count:Q", title="Applications", format=","),
                alt.Tooltip("pct:Q", title="Portfolio share (%)", format=".1f"),
            ],
        )
        .properties(
            **_single_chart_props(
                _chart_height(compact=compact), "Risk Segment Volume", compact=compact
            )
        )
    )


def driver_importance_bars(features: list[dict[str, Any]], *, title: str, color: str) -> alt.Chart:
    if not features:
        frame = pd.DataFrame({"label": ["—"], "importance": [0.0]})
    else:
        frame = pd.DataFrame(features)
        frame["label"] = frame["feature"].astype(str)

    label_limit = 28
    frame["label_short"] = frame["label"].apply(
        lambda s: s if len(str(s)) <= label_limit else str(s)[: label_limit - 1] + "…"
    )

    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y(
                "label_short:N",
                sort=alt.EncodingSortField(field="importance", order="descending"),
                title=None,
                axis=alt.Axis(labelLimit=200),
            ),
            x=alt.X("importance:Q", title="Mean |SHAP|", axis=alt.Axis(format=".3f")),
            color=alt.value(color),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(
            height=max(180, 40 * len(frame)),
            title=_chart_title(title),
            padding=_chart_padding(),
        )
    )


def build_chart_safe(
    primary_builder: Callable[..., alt.Chart],
    fallback_builder: Callable[..., alt.Chart],
    *args: Any,
    primary_kwargs: dict[str, Any] | None = None,
    fallback_kwargs: dict[str, Any] | None = None,
) -> alt.Chart:
    """Return primary chart, or fallback if construction fails."""
    pk = primary_kwargs or {}
    fk = fallback_kwargs or {}
    try:
        return primary_builder(*args, **pk)
    except Exception:
        return fallback_builder(*args, **fk)
