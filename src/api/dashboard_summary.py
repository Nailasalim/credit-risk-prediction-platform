"""
Executive dashboard payload — business metrics from portfolio scoring + SHAP.
"""

from __future__ import annotations

from typing import Any

from src.ml.global_importance import load_global_importance
from src.ml.portfolio_analytics import get_portfolio_analytics


def build_dashboard_summary(top_features: int = 6) -> dict[str, Any]:
    portfolio = get_portfolio_analytics()
    importance = load_global_importance()

    top = importance.head(top_features)
    drivers = [
        {
            "feature": row["feature"],
            "importance": round(float(row["importance"]), 6),
            "display_impact": round(float(row["display_impact"]), 6),
            "direction": "increases risk" if float(row["display_impact"]) > 0 else "decreases risk",
        }
        for _, row in top.iterrows()
    ]

    executive_kpis = {
        "total_applications": int(portfolio["scored_records"]),
        "default_rate_pct": float(portfolio["observed_default_rate_pct"]),
        "approval_rate_pct": float(portfolio["approval_rate_pct"]),
        "high_risk_pct": float(portfolio["high_risk_pct"]),
        "high_risk_count": int(portfolio["high_risk_count"]),
    }

    return {
        "executive_kpis": executive_kpis,
        "portfolio": portfolio,
        "top_risk_drivers": {
            "source": str(importance["source"].iloc[0]) if not importance.empty else "unknown",
            "shap_sample_count": int(importance["sample_count"].iloc[0]) if not importance.empty else 0,
            "features": drivers,
        },
        "risk_band_distribution": portfolio["risk_band_distribution"],
        "scoring_runtime_seconds": portfolio.get("scoring_runtime_seconds"),
    }
