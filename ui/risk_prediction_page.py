"""
Risk Prediction page — applicant form and live /decision API integration.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = os.environ.get("CREDIT_RISK_API_URL", "http://127.0.0.1:8000")
DECISION_PATH = "/decision"

DEFAULT_APPLICANT: dict[str, Any] = {
    "EXT_SOURCE_1": 0.52,
    "EXT_SOURCE_2": 0.61,
    "EXT_SOURCE_3": 0.68,
    "AMT_INCOME_TOTAL": 864_000.0,
    "AMT_CREDIT": 450_000.0,
    "AMT_ANNUITY": 18_500.0,
    "AMT_GOODS_PRICE": 450_000.0,
    "DAYS_BIRTH": -12_500,
    "DAYS_EMPLOYED": -1_460,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "DAYS_LAST_PHONE_CHANGE": -1_100,
    "DAYS_ID_PUBLISH": -2_900,
    "REG_CITY_NOT_WORK_CITY": 0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "FLAG_EMP_PHONE": 1,
    "FLAG_DOCUMENT_3": 1,
    "OWN_CAR_AGE": 2.0,
}

REGION_LABELS = {
    1: "1 — Favourable",
    2: "2 — Standard",
    3: "3 — Elevated risk",
}

BAND_STYLES: dict[str, dict[str, str]] = {
    "LOW": {
        "label": "Low Risk",
        "color": "#22c55e",
        "bg": "rgba(34, 197, 94, 0.12)",
        "border": "rgba(34, 197, 94, 0.35)",
    },
    "MEDIUM": {
        "label": "Medium Risk",
        "color": "#f59e0b",
        "bg": "rgba(245, 158, 11, 0.12)",
        "border": "rgba(245, 158, 11, 0.35)",
    },
    "HIGH": {
        "label": "High Risk",
        "color": "#ef4444",
        "bg": "rgba(239, 68, 68, 0.12)",
        "border": "rgba(239, 68, 68, 0.35)",
    },
}

RECOMMENDATION_STYLES: dict[str, dict[str, str]] = {
    "APPROVE": {"color": "#22c55e", "icon": "✓", "label": "Approve"},
    "REVIEW": {"color": "#f59e0b", "icon": "◐", "label": "Manual review"},
    "DECLINE": {"color": "#ef4444", "icon": "✕", "label": "Decline"},
}


def _years_from_days_employed(days: int) -> float:
    return round(abs(days) / 365.25, 1)


def _days_employed_from_years(years: float) -> int:
    return -int(years * 365.25)


def _format_amount(value: float) -> str:
    """Display whole rupee amounts without decimals."""
    return f"{int(round(value)):,}"


def _parse_amount(text: str, fallback: float) -> float:
    cleaned = text.replace(",", "").replace("₹", "").strip()
    if not cleaned:
        return fallback
    return float(cleaned)


def _parse_decimal(text: str, fallback: float, *, min_val: float | None = None, max_val: float | None = None) -> float:
    cleaned = text.strip()
    if not cleaned:
        return fallback
    value = float(cleaned)
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


def _parse_int(text: str, fallback: int) -> int:
    cleaned = text.replace(",", "").strip()
    if not cleaned:
        return fallback
    return int(float(cleaned))


def money_input(label: str, default: float, key: str) -> float:
    """Plain text field for currency — no +/- steppers."""
    raw = st.text_input(label, value=_format_amount(default), key=key)
    return _parse_amount(raw, default)


def decimal_input(
    label: str,
    default: float,
    key: str,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    placeholder: str = "",
) -> float:
    raw = st.text_input(
        label,
        value=f"{default:.2f}",
        key=key,
        placeholder=placeholder,
    )
    return _parse_decimal(raw, default, min_val=min_val, max_val=max_val)


def plain_number_input(label: str, default: float, key: str, *, decimals: bool = True) -> float:
    display = str(default) if decimals else str(int(default))
    raw = st.text_input(label, value=display, key=key)
    return _parse_decimal(raw, default) if decimals else float(_parse_int(raw, int(default)))


def integer_input(label: str, default: int, key: str) -> int:
    raw = st.text_input(label, value=str(default), key=key)
    return _parse_int(raw, default)


def yes_no_select(label: str, default: int, key: str) -> int:
    return st.selectbox(
        label,
        options=[0, 1],
        index=int(default),
        format_func=lambda x: "Yes" if x else "No",
        key=key,
    )


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def inject_risk_prediction_styles() -> None:
    st.markdown(
        """
        <style>
        .rp-form-panel {
            background: #161d27;
            border: 1px solid #243044;
            border-radius: 12px;
            padding: 1rem 1.15rem 0.25rem 1.15rem;
        }
        .rp-section-title {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b7c96;
            margin: 0.5rem 0 0.45rem 0;
        }
        .rp-result-shell {
            background: #161d27;
            border: 1px solid #243044;
            border-radius: 12px;
            padding: 1rem 1.15rem;
        }
        .rp-placeholder {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.85rem 1rem;
            background: #121820;
            border: 1px dashed #2a3a52;
            border-radius: 10px;
            color: #8b9bb4;
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .rp-placeholder-icon {
            font-size: 1.25rem;
            opacity: 0.7;
            flex-shrink: 0;
        }
        .rp-status {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            margin-bottom: 0.75rem;
        }
        .rp-status.success { background: rgba(34,197,94,0.12); color: #4ade80; }
        .rp-status.error { background: rgba(239,68,68,0.12); color: #f87171; }

        .rp-badge {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
        }
        .rp-gauge-wrap { text-align: center; margin: 0.25rem 0 0.85rem 0; }
        .rp-gauge {
            width: 180px; height: 90px; margin: 0 auto; position: relative; overflow: hidden;
        }
        .rp-gauge-bg {
            width: 180px; height: 180px; border-radius: 50%;
            background: conic-gradient(
                from 180deg,
                var(--gauge-color) 0deg,
                var(--gauge-color) calc(var(--gauge-pct) * 1.8deg),
                #243044 calc(var(--gauge-pct) * 1.8deg),
                #243044 180deg
            );
            clip-path: inset(50% 0 0 0);
        }
        .rp-gauge-inner {
            position: absolute; left: 50%; bottom: 0; transform: translateX(-50%);
            width: 126px; height: 63px; background: #121820;
            border-radius: 126px 126px 0 0;
            display: flex; align-items: flex-end; justify-content: center;
            padding-bottom: 0.3rem;
        }
        .rp-gauge-score { font-size: 1.75rem; font-weight: 800; color: #e8edf5; line-height: 1; }
        .rp-gauge-label {
            font-size: 0.68rem; color: #6b7c96; text-transform: uppercase;
            letter-spacing: 0.06em; margin-top: 0.2rem;
        }
        .rp-metric-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem;
        }
        .rp-metric {
            background: #0f141c; border: 1px solid #243044;
            border-radius: 8px; padding: 0.65rem 0.75rem;
        }
        .rp-metric-label {
            font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.05em; color: #6b7c96; margin: 0 0 0.2rem 0;
        }
        .rp-metric-value { font-size: 1.05rem; font-weight: 700; color: #e8edf5; margin: 0; }
        .rp-confidence-bar {
            height: 5px; background: #243044; border-radius: 999px;
            overflow: hidden; margin-top: 0.3rem;
        }
        .rp-confidence-fill {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
        }
        .rp-recommendation {
            margin-top: 0.75rem; padding: 0.8rem 0.9rem;
            border-radius: 8px; border: 1px solid #243044; background: #0f141c;
        }
        .rp-recommendation-title {
            font-size: 0.65rem; font-weight: 600; text-transform: uppercase;
            color: #6b7c96; margin: 0 0 0.25rem 0;
        }
        .rp-recommendation-value { font-size: 1.1rem; font-weight: 800; margin: 0; }
        .rp-col-heading {
            font-size: 0.95rem; font-weight: 600; color: #c5d0e0;
            margin: 0 0 0.65rem 0;
        }
        .rp-adv-title {
            font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.08em; color: #5c6d86;
            margin: 1rem 0 0.5rem 0; padding-top: 0.5rem;
            border-top: 1px solid #243044;
        }
        .rp-adv-title:first-of-type {
            margin-top: 0.25rem; padding-top: 0; border-top: none;
        }
        /* Banking-style fields: hide number-input steppers if any remain */
        div[data-testid="stNumberInput"] button {
            display: none !important;
        }
        div[data-testid="stNumberInput"] input {
            -moz-appearance: textfield;
        }
        div[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
        div[data-testid="stNumberInput"] input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


class DecisionAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def post_decision(api_base: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    url = api_base.rstrip("/") + DECISION_PATH
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw), (time.perf_counter() - start) * 1000
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            message = detail or exc.reason
        raise DecisionAPIError(str(message), status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise DecisionAPIError(
            f"Cannot reach API at {url}. Is uvicorn running? ({exc.reason})"
        ) from exc


def build_model_payload(form: dict[str, Any]) -> dict[str, Any]:
    """Map form values to API field names (unchanged backend contract)."""
    return {
        "EXT_SOURCE_1": float(form["ext_source_1"]),
        "EXT_SOURCE_2": float(form["ext_source_2"]),
        "EXT_SOURCE_3": float(form["ext_source_3"]),
        "AMT_INCOME_TOTAL": float(form["amt_income_total"]),
        "AMT_CREDIT": float(form["amt_credit"]),
        "AMT_ANNUITY": float(form["amt_annuity"]),
        "AMT_GOODS_PRICE": float(form["amt_goods_price"]),
        "DAYS_BIRTH": int(form["days_birth"]),
        "DAYS_EMPLOYED": int(form["days_employed"]),
        "REGION_RATING_CLIENT": int(form["region_rating_client"]),
        "REGION_RATING_CLIENT_W_CITY": int(form["region_rating_client_w_city"]),
        "DAYS_LAST_PHONE_CHANGE": int(form["days_last_phone_change"]),
        "DAYS_ID_PUBLISH": int(form["days_id_publish"]),
        "REG_CITY_NOT_WORK_CITY": int(form["reg_city_not_work_city"]),
        "REG_CITY_NOT_LIVE_CITY": int(form["reg_city_not_live_city"]),
        "FLAG_EMP_PHONE": int(form["flag_emp_phone"]),
        "FLAG_DOCUMENT_3": int(form["flag_document_3"]),
        "OWN_CAR_AGE": float(form["own_car_age"]),
    }


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------


def _band_style(risk_band: str) -> dict[str, str]:
    return BAND_STYLES.get(risk_band.upper(), BAND_STYLES["MEDIUM"])


def render_compact_placeholder() -> None:
    st.markdown(
        """
        <div class="rp-placeholder">
            <span class="rp-placeholder-icon">📋</span>
            <span>Run a risk assessment to view results.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_gauge(risk_score: int, risk_band: str) -> None:
    style = _band_style(risk_band)
    score = max(0, min(100, int(risk_score)))
    st.markdown(
        f"""
        <div class="rp-gauge-wrap">
            <div class="rp-gauge">
                <div class="rp-gauge-bg" style="--gauge-pct:{score};--gauge-color:{style['color']};"></div>
                <div class="rp-gauge-inner"><span class="rp-gauge-score">{score}</span></div>
            </div>
            <p class="rp-gauge-label">Risk score</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_band: str) -> None:
    style = _band_style(risk_band)
    st.markdown(
        f'<span class="rp-badge" style="color:{style["color"]};background:{style["bg"]};'
        f'border:1px solid {style["border"]};">{style["label"]}</span>',
        unsafe_allow_html=True,
    )


def render_result_card(result: dict[str, Any], elapsed_ms: float) -> None:
    risk_band = str(result.get("risk_band", "MEDIUM")).upper()
    risk_score = int(result.get("risk_score", 0))
    default_pct = float(result.get("default_probability", 0))
    approval_pct = float(result.get("approval_probability", 0))
    recommendation = str(result.get("recommendation", "REVIEW")).upper()
    confidence = int(result.get("confidence", 0))
    rec = RECOMMENDATION_STYLES.get(recommendation, RECOMMENDATION_STYLES["REVIEW"])

    st.markdown(
        f'<span class="rp-status success">✓ Scored in {elapsed_ms:.0f} ms</span>',
        unsafe_allow_html=True,
    )
    render_risk_badge(risk_band)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    render_risk_gauge(risk_score, risk_band)

    st.markdown(
        f"""
        <div class="rp-metric-grid">
            <div class="rp-metric">
                <p class="rp-metric-label">Default probability</p>
                <p class="rp-metric-value" style="color:#f59e0b;">{default_pct:.1f}%</p>
            </div>
            <div class="rp-metric">
                <p class="rp-metric-label">Approval probability</p>
                <p class="rp-metric-value" style="color:#22c55e;">{approval_pct:.1f}%</p>
            </div>
            <div class="rp-metric" style="grid-column: span 2;">
                <p class="rp-metric-label">Confidence</p>
                <p class="rp-metric-value">{confidence}%</p>
                <div class="rp-confidence-bar">
                    <div class="rp-confidence-fill" style="width:{confidence}%;"></div>
                </div>
            </div>
        </div>
        <div class="rp-recommendation" style="border-color:{rec['color']}33;">
            <p class="rp-recommendation-title">Recommendation</p>
            <p class="rp-recommendation-value" style="color:{rec['color']};">
                {rec['icon']} {rec['label']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_advanced_inputs(
    defaults: dict[str, Any],
    region_rating: int | None = None,
) -> dict[str, Any]:
    """Secondary model fields — grouped for underwriter review."""
    st.caption("Optional details. Defaults are suitable for most assessments.")

    city_default = region_rating if region_rating is not None else int(defaults["REGION_RATING_CLIENT_W_CITY"])

    st.markdown('<p class="rp-adv-title">Demographics</p>', unsafe_allow_html=True)
    demo1, demo2 = st.columns(2)
    with demo1:
        age_years = plain_number_input(
            "Applicant age",
            float(int(abs(defaults["DAYS_BIRTH"]) / 365.25)),
            key="adv_age",
            decimals=False,
        )
    with demo2:
        own_car_age = plain_number_input(
            "Vehicle age",
            float(defaults["OWN_CAR_AGE"]),
            key="adv_vehicle_age",
        )

    st.markdown('<p class="rp-adv-title">Location</p>', unsafe_allow_html=True)
    region_city = st.selectbox(
        "Region & city rating",
        options=[1, 2, 3],
        index=city_default - 1,
        format_func=lambda x: REGION_LABELS[x],
        key="adv_region_city",
    )
    loc1, loc2 = st.columns(2)
    with loc1:
        reg_not_work = yes_no_select(
            "Registered city different from work city",
            int(defaults["REG_CITY_NOT_WORK_CITY"]),
            key="adv_reg_work",
        )
    with loc2:
        reg_not_live = yes_no_select(
            "Registered city different from residence city",
            int(defaults["REG_CITY_NOT_LIVE_CITY"]),
            key="adv_reg_live",
        )

    st.markdown('<p class="rp-adv-title">Identity & activity</p>', unsafe_allow_html=True)
    id1, id2 = st.columns(2)
    with id1:
        days_id = integer_input(
            "Days since ID issued",
            int(defaults["DAYS_ID_PUBLISH"]),
            key="adv_days_id",
        )
        flag_emp_phone = yes_no_select(
            "Employer phone available",
            int(defaults["FLAG_EMP_PHONE"]),
            key="adv_emp_phone",
        )
    with id2:
        days_phone = integer_input(
            "Days since phone change",
            int(defaults["DAYS_LAST_PHONE_CHANGE"]),
            key="adv_days_phone",
        )
        flag_doc3 = yes_no_select(
            "ID document verified",
            int(defaults["FLAG_DOCUMENT_3"]),
            key="adv_id_doc",
        )

    return {
        "days_birth": -int(age_years * 365.25),
        "region_rating_client_w_city": region_city,
        "days_last_phone_change": days_phone,
        "days_id_publish": days_id,
        "reg_city_not_work_city": reg_not_work,
        "reg_city_not_live_city": reg_not_live,
        "flag_emp_phone": flag_emp_phone,
        "flag_document_3": flag_doc3,
        "own_car_age": own_car_age,
    }


RISK_FORM_WIDGET_KEYS: tuple[str, ...] = (
    "applicant_name",
    "applicant_id",
    "fin_income",
    "fin_credit",
    "fin_annuity",
    "fin_goods",
    "score_ext1",
    "score_ext2",
    "score_ext3",
    "emp_years",
    "main_region",
    "adv_age",
    "adv_vehicle_age",
    "adv_region_city",
    "adv_reg_work",
    "adv_reg_live",
    "adv_days_id",
    "adv_emp_phone",
    "adv_days_phone",
    "adv_id_doc",
)


def _clear_risk_form_state() -> None:
    """Reset form widgets and assessment results."""
    for key in RISK_FORM_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state.risk_result = None
    st.session_state.risk_status = "idle"
    st.session_state.risk_elapsed_ms = 0.0
    st.session_state.risk_error = None
    st.session_state.pop("last_applicant_payload", None)
    st.session_state.pop("applicant_name", None)
    st.session_state.pop("applicant_id", None)


def render_applicant_form() -> dict[str, Any] | None:
    """Applicant form — primary fields plus optional advanced inputs."""
    defaults = DEFAULT_APPLICANT

    with st.form("risk_prediction_form", clear_on_submit=False):
        st.markdown('<div class="rp-form-panel">', unsafe_allow_html=True)

        st.markdown('<p class="rp-section-title">Applicant identity</p>', unsafe_allow_html=True)
        id1, id2 = st.columns(2)
        with id1:
            applicant_name = st.text_input(
                "Applicant name",
                value=st.session_state.get("applicant_name", ""),
                placeholder="e.g. Priya Nair",
                key="applicant_name",
            )
        with id2:
            applicant_id = st.text_input(
                "Applicant ID",
                value=st.session_state.get("applicant_id", ""),
                placeholder="e.g. APP-100291",
                key="applicant_id",
            )

        st.markdown('<p class="rp-section-title">Financial profile</p>', unsafe_allow_html=True)
        fin1, fin2 = st.columns(2)
        with fin1:
            amt_income = money_input("Annual income", float(defaults["AMT_INCOME_TOTAL"]), "fin_income")
            amt_credit = money_input("Credit amount", float(defaults["AMT_CREDIT"]), "fin_credit")
        with fin2:
            amt_annuity = money_input(
                "Monthly annuity (₹/mo)",
                float(defaults["AMT_ANNUITY"]),
                "fin_annuity",
            )
            amt_goods = money_input("Goods price", float(defaults["AMT_GOODS_PRICE"]), "fin_goods")

        st.markdown('<p class="rp-section-title">Bureau scores</p>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            ext1 = decimal_input(
                "External credit score 1",
                float(defaults["EXT_SOURCE_1"]),
                "score_ext1",
                min_val=0.0,
                max_val=1.0,
                placeholder="0.00 – 1.00",
            )
        with sc2:
            ext2 = decimal_input(
                "External credit score 2",
                float(defaults["EXT_SOURCE_2"]),
                "score_ext2",
                min_val=0.0,
                max_val=1.0,
                placeholder="0.00 – 1.00",
            )
        with sc3:
            ext3 = decimal_input(
                "External credit score 3",
                float(defaults["EXT_SOURCE_3"]),
                "score_ext3",
                min_val=0.0,
                max_val=1.0,
                placeholder="0.00 – 1.00",
            )

        st.markdown('<p class="rp-section-title">Employment & region</p>', unsafe_allow_html=True)
        emp1, emp2 = st.columns(2)
        with emp1:
            years_employed = plain_number_input(
                "Employment years",
                _years_from_days_employed(int(defaults["DAYS_EMPLOYED"])),
                "emp_years",
            )
        with emp2:
            region = st.selectbox(
                "Region rating",
                options=[1, 2, 3],
                index=int(defaults["REGION_RATING_CLIENT"]) - 1,
                format_func=lambda x: REGION_LABELS[x],
                key="main_region",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Advanced inputs", expanded=False):
            advanced = _render_advanced_inputs(defaults, region_rating=region)

        btn_run, btn_clear = st.columns(2)
        with btn_run:
            submitted = st.form_submit_button(
                "Run Risk Assessment",
                type="primary",
                use_container_width=True,
            )
        with btn_clear:
            cleared = st.form_submit_button("Clear form", use_container_width=True)

    if cleared:
        _clear_risk_form_state()
        st.rerun()

    if not submitted:
        return None

    return {
        "applicant_name": applicant_name,
        "applicant_id": applicant_id,
        "ext_source_1": ext1,
        "ext_source_2": ext2,
        "ext_source_3": ext3,
        "amt_income_total": amt_income,
        "amt_credit": amt_credit,
        "amt_annuity": amt_annuity,
        "amt_goods_price": amt_goods,
        "days_birth": advanced["days_birth"],
        "days_employed": _days_employed_from_years(years_employed),
        "region_rating_client": region,
        "region_rating_client_w_city": advanced["region_rating_client_w_city"],
        "days_last_phone_change": advanced["days_last_phone_change"],
        "days_id_publish": advanced["days_id_publish"],
        "reg_city_not_work_city": advanced["reg_city_not_work_city"],
        "reg_city_not_live_city": advanced["reg_city_not_live_city"],
        "flag_emp_phone": advanced["flag_emp_phone"],
        "flag_document_3": advanced["flag_document_3"],
        "own_car_age": advanced["own_car_age"],
    }


def render_risk_prediction_page(page_header_fn: Callable[..., None]) -> None:
    inject_risk_prediction_styles()

    page_header_fn(
        "Risk Prediction",
        "Capture applicant details and receive an instant credit risk decision.",
        badge="Risk Engine",
    )

    api_base = st.session_state.get("api_base_url", DEFAULT_API_BASE)
    with st.expander("API connection", expanded=False):
        api_base = st.text_input("Backend URL", value=api_base)
        st.session_state["api_base_url"] = api_base

    for key in ("risk_result", "risk_status", "risk_elapsed_ms", "risk_error"):
        if key not in st.session_state:
            st.session_state[key] = None if key == "risk_result" else ("idle" if key == "risk_status" else 0.0 if key == "risk_elapsed_ms" else None)

    form_col, result_col = st.columns(2, gap="large")

    with form_col:
        st.markdown('<p class="rp-col-heading">Applicant form</p>', unsafe_allow_html=True)
        form_data = render_applicant_form()

    with result_col:
        st.markdown('<p class="rp-col-heading">Risk assessment results</p>', unsafe_allow_html=True)
        st.markdown('<div class="rp-result-shell">', unsafe_allow_html=True)

        if form_data is not None:
            try:
                with st.spinner("Scoring…"):
                    model_payload = build_model_payload(form_data)
                    result, elapsed = post_decision(api_base, model_payload)
                from assessment_store import record_assessment

                record_assessment(
                    str(form_data.get("applicant_name", "")),
                    str(form_data.get("applicant_id", "")),
                    model_payload,
                    result,
                )
                st.session_state.risk_elapsed_ms = elapsed
                st.session_state.risk_status = "success"
                st.session_state.risk_error = None
            except DecisionAPIError as exc:
                st.session_state.risk_status = "error"
                st.session_state.risk_error = str(exc)
                st.session_state.risk_result = None
            except Exception as exc:
                st.session_state.risk_status = "error"
                st.session_state.risk_error = f"Unexpected error: {exc}"
                st.session_state.risk_result = None

        status = st.session_state.risk_status
        if status == "error":
            st.markdown('<span class="rp-status error">✕ Assessment failed</span>', unsafe_allow_html=True)
            st.error(st.session_state.risk_error or "Request failed.")
        elif status == "success" and st.session_state.risk_result:
            render_result_card(st.session_state.risk_result, st.session_state.risk_elapsed_ms)
            if st.button("View Explanation →", use_container_width=True, key="goto_explainability"):
                from app_navigation import navigate_to_explainability_latest

                navigate_to_explainability_latest()
                st.rerun()
        else:
            render_compact_placeholder()

        st.markdown("</div>", unsafe_allow_html=True)
