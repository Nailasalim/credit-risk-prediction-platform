"""
Decision Rules page — enterprise underwriting rules catalog (GET /rules).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

import streamlit as st

DEFAULT_API_BASE = os.environ.get("CREDIT_RISK_API_URL", "http://127.0.0.1:8000")
RULES_PATH = "/rules"

FILTER_ALL   = "All"
FILTER_HIGH  = "High Risk"
FILTER_MED   = "Medium Risk"
FILTER_LOW   = "Low Risk"

FILTER_TO_BAND = {FILTER_HIGH: "HIGH", FILTER_MED: "MEDIUM", FILTER_LOW: "LOW"}

BAND_CFG = {
    "HIGH":   {"label": "HIGH RISK",   "color": "#f87171", "bg": "rgba(239,68,68,0.12)",  "border": "#ef4444"},
    "MEDIUM": {"label": "MEDIUM RISK", "color": "#fbbf24", "bg": "rgba(245,158,11,0.12)", "border": "#f59e0b"},
    "LOW":    {"label": "LOW RISK",    "color": "#4ade80", "bg": "rgba(34,197,94,0.12)",  "border": "#22c55e"},
}

ACTION_CFG = {
    "DECLINE": {"label": "Decline",       "color": "#f87171", "bg": "rgba(239,68,68,0.12)",  "icon": "✕"},
    "REVIEW":  {"label": "Manual Review", "color": "#fbbf24", "bg": "rgba(245,158,11,0.12)", "icon": "⊙"},
    "APPROVE": {"label": "Approve",       "color": "#4ade80", "bg": "rgba(34,197,94,0.12)",  "icon": "✓"},
}


class RulesAPIError(Exception):
    pass


def fetch_rules_catalog(api_base: str) -> dict[str, Any]:
    url = api_base.rstrip("/") + RULES_PATH
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            msg = json.loads(raw).get("detail", raw)
        except json.JSONDecodeError:
            msg = raw or exc.reason
        raise RulesAPIError(str(msg)) from exc
    except urllib.error.URLError as exc:
        raise RulesAPIError(f"Cannot reach API at {url}. Is uvicorn running? ({exc.reason})") from exc


def _merge_session_matches(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matched  = st.session_state.get("risk_result", {}) or {}
    fired    = matched.get("matched_rules") or []
    fired_ids = {r.get("rule_id") for r in fired if r.get("rule_id")}
    return [{**rule, "matched": rule.get("rule_id") in fired_ids} for rule in rules]


def _confidence_pct(rule: dict[str, Any]) -> float:
    if "confidence_pct" in rule:
        return float(rule["confidence_pct"])
    c = float(rule.get("confidence", 0))
    return c * 100.0 if c <= 1.0 else c


def _coverage_pct(rule: dict[str, Any]) -> float:
    if "coverage" in rule:
        return float(rule["coverage"]) * 100.0
    sup = int(rule.get("support", 0))
    port = float(rule.get("portfolio_size", 0))
    return (sup / port * 100.0) if port > 0 else 0.0


def _condition_labels(rule: dict[str, Any]) -> list[str]:
    lbl = rule.get("condition_labels")
    if lbl:
        return list(lbl)
    return [str(c) for c in (rule.get("conditions") or [])]


# ── CSS ──────────────────────────────────────────────────────────────────────

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

/* meta row */
.dr-meta {
    display:flex; align-items:center; gap:1.2rem;
    font-size:0.72rem; color:#3d5270; margin-bottom:1rem;
}
.dr-live-dot {
    width:6px; height:6px; border-radius:50%;
    background:#22c55e; display:inline-block;
    margin-right:0.3rem;
    box-shadow:0 0 6px rgba(34,197,94,0.55);
}

/* KPI strip */
.dr-kpi {
    background:#111720; border:1px solid #1e2d42;
    border-radius:9px; padding:0.75rem 1rem;
    position:relative; overflow:hidden;
    text-align:left;
}
.dr-kpi-bar {
    position:absolute; top:0; left:0; right:0;
    height:2px; border-radius:9px 9px 0 0;
}
.dr-kpi-lbl {
    font-size:0.6rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.08em; color:#3d5270; margin:0 0 0.35rem 0;
    font-family:'DM Sans',sans-serif;
}
.dr-kpi-val {
    font-family:'JetBrains Mono',monospace;
    font-size:1.7rem; font-weight:600; color:#e8edf5;
    margin:0; line-height:1;
}
.dr-kpi-val.c-high   { color:#f87171; }
.dr-kpi-val.c-medium { color:#fbbf24; }
.dr-kpi-val.c-low    { color:#4ade80; }
.dr-kpi-sub {
    font-size:0.6rem; color:#2a3f5c; margin:0.3rem 0 0 0;
    text-transform:uppercase; letter-spacing:0.05em;
    font-family:'DM Sans',sans-serif;
}

/* triggered panel */
.dr-trig {
    display:flex; align-items:flex-start; gap:0.6rem;
    background:rgba(59,130,246,0.06);
    border:1px solid rgba(59,130,246,0.2);
    border-left:3px solid #3b82f6;
    border-radius:8px; padding:0.65rem 0.9rem;
    margin-bottom:0.9rem;
    font-size:0.77rem; color:#7a9cc4;
    max-width:920px;
}
.dr-trig b { color:#93c5fd; font-weight:600; }

/* filter label */
.dr-filter-lbl {
    font-size:0.62rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.07em; color:#3d5270; margin-bottom:0.3rem;
    font-family:'DM Sans',sans-serif;
}

/* radio pill overrides */
div[data-testid="stRadio"] > div {
    flex-direction:row !important; flex-wrap:wrap; gap:0.3rem !important;
}
div[data-testid="stRadio"] label {
    background:#111720 !important;
    border:1px solid #1e2d42 !important;
    border-radius:6px !important;
    padding:0.28rem 0.75rem !important;
    font-size:0.73rem !important;
    font-family:'DM Sans',sans-serif !important;
    color:#4e6280 !important;
    transition:all 0.15s;
}
div[data-testid="stRadio"] label:hover {
    border-color:#2a3f5c !important; color:#7a9cc4 !important;
}
div[data-testid="stRadio"] label[data-checked="true"] {
    border-color:#3b82f6 !important;
    background:rgba(59,130,246,0.12) !important;
    color:#93c5fd !important;
}
div[data-testid="stRadio"] > label { display:none; }

/* empty state */
.dr-empty {
    text-align:center; padding:2.5rem 1rem;
    color:#2e4060; font-size:0.82rem;
    font-family:'DM Sans',sans-serif;
}
</style>
"""

# ── Rule card rendered via st.container() + st.markdown (flat HTML only) ────

CARD_CSS = """
<style>
.rcard {
    background:#111720;
    border:1px solid #1c2a3a;
    border-left:3px solid VAR_STRIPE;
    border-radius:10px;
    padding:0.85rem 1rem 0.75rem 1rem;
    margin-bottom:0.5rem;
    max-width:920px;
    font-family:'DM Sans',sans-serif;
}
.rcard.is-matched {
    border-color:rgba(59,130,246,0.4);
    border-left-color:#3b82f6;
    box-shadow:0 0 0 1px rgba(59,130,246,0.12);
}
.rcard-hd {
    display:flex; align-items:center;
    gap:0.5rem; margin-bottom:0.65rem; flex-wrap:wrap;
}
.rcard-band {
    font-size:0.58rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.07em;
    padding:0.2rem 0.48rem; border-radius:4px;
}
.rcard-title {
    font-size:0.88rem; font-weight:600;
    color:#cdd6e8; margin:0; flex:1; line-height:1.35;
}
.rcard-matched {
    font-size:0.58rem; font-weight:700; letter-spacing:0.06em;
    color:#60a5fa; background:rgba(59,130,246,0.12);
    border:1px solid rgba(59,130,246,0.25);
    padding:0.16rem 0.4rem; border-radius:4px;
}
.rcard-sec {
    font-size:0.6rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.08em; color:#3d5270; margin:0 0 0.28rem 0;
}
.rcard-cond {
    font-family:'JetBrains Mono',monospace;
    font-size:0.72rem; color:#6b8db5;
    padding:0.15rem 0; border-bottom:1px solid #131f2e;
    display:flex; align-items:center; gap:0.35rem;
}
.rcard-cond:last-child { border-bottom:none; }
.rcard-cond-arrow { color:#2a4060; }
.rcard-action {
    display:inline-flex; align-items:center; gap:0.3rem;
    font-size:0.73rem; font-weight:700;
    padding:0.28rem 0.65rem; border-radius:6px;
    margin-top:0.25rem;
}
.rcard-reason {
    margin:0.55rem 0 0 0; padding-top:0.5rem;
    border-top:1px solid #131f2e;
    line-height:1.55;
}
.rcard-reason-lbl {
    font-size:0.6rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.08em; color:#3d5270; margin-right:0.45rem;
}
.rcard-reason-text {
    font-size:0.76rem; color:#7a9cc4; font-style:normal;
}
.rcard-footer {
    display:flex; flex-wrap:wrap; gap:0 1.4rem;
    padding-top:0.5rem; margin-top:0.5rem;
    border-top:1px solid #131f2e;
}
.rcard-metric {
    display:flex; align-items:center; gap:0.3rem;
    font-size:0.68rem;
}
.rcard-mlbl {
    color:#2e4060; font-weight:600; text-transform:uppercase;
    letter-spacing:0.05em; font-size:0.59rem;
}
.rcard-mval {
    font-family:'JetBrains Mono',monospace;
    font-size:0.71rem; color:#5a7ea8; font-weight:500;
}
.rcard-track {
    width:64px; height:3px; background:#1a2535;
    border-radius:2px; overflow:hidden; display:inline-block;
    vertical-align:middle;
}
.rcard-fill {
    height:100%; border-radius:2px;
    background:linear-gradient(90deg,#3b82f6,#60a5fa);
}
</style>
"""


def inject_styles() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    st.markdown(CARD_CSS, unsafe_allow_html=True)


# ── Render helpers ────────────────────────────────────────────────────────────

def render_summary_kpis(rules: list[dict[str, Any]]) -> None:
    total  = len(rules)
    high   = sum(1 for r in rules if str(r.get("risk_band","")).upper()=="HIGH")
    medium = sum(1 for r in rules if str(r.get("risk_band","")).upper()=="MEDIUM")
    low    = sum(1 for r in rules if str(r.get("risk_band","")).upper()=="LOW")
    fired  = sum(1 for r in rules if r.get("matched"))

    kpis = [
        ("Active Rules",   str(total),  "",         "linear-gradient(90deg,#3b82f6,#6366f1)", f"{fired} triggered"),
        ("High Risk",      str(high),   "c-high",   "linear-gradient(90deg,#ef4444,#f97316)", "decline threshold"),
        ("Medium Risk",    str(medium), "c-medium", "linear-gradient(90deg,#f59e0b,#eab308)", "review threshold"),
        ("Low Risk",       str(low),    "c-low",    "linear-gradient(90deg,#22c55e,#10b981)", "approve threshold"),
    ]
    cols = st.columns(4)
    for col, (lbl, val, css, grad, sub) in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="dr-kpi">'
                f'<div class="dr-kpi-bar" style="background:{grad}"></div>'
                f'<p class="dr-kpi-lbl">{lbl}</p>'
                f'<p class="dr-kpi-val {css}">{val}</p>'
                f'<p class="dr-kpi-sub">{sub}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_rule_card(rule: dict[str, Any]) -> None:
    band     = str(rule.get("risk_band", "MEDIUM")).upper()
    bc       = BAND_CFG.get(band, BAND_CFG["MEDIUM"])
    action   = str(rule.get("action", "REVIEW")).upper()
    ac       = ACTION_CFG.get(action, ACTION_CFG["REVIEW"])
    matched  = bool(rule.get("matched"))
    conds    = _condition_labels(rule)
    conf     = _confidence_pct(rule)
    cov      = _coverage_pct(rule)
    rationale = rule.get("rationale") or ""

    matched_cls = " is-matched" if matched else ""
    matched_tag = '<span class="rcard-matched">▶ TRIGGERED</span>' if matched else ""

    # ── Build card HTML in flat pieces (no nested f-string interpolation) ────

    html_parts: list[str] = []

    # open card
    html_parts.append(
        f'<div class="rcard{matched_cls}" style="border-left-color:{bc["border"]}">'
    )

    # header
    html_parts.append('<div class="rcard-hd">')
    html_parts.append(
        f'<span class="rcard-band" style="color:{bc["color"]};'
        f'background:{bc["bg"]};border:1px solid {bc["border"]}">'
        f'{bc["label"]}</span>'
    )
    html_parts.append(
        f'<span class="rcard-title">{rule.get("title","Rule")}{matched_tag}</span>'
    )
    html_parts.append('</div>')

    # two-column layout using a simple flex row
    html_parts.append(
        '<div style="display:flex;align-items:flex-start;'
        'gap:1.4rem;justify-content:space-between">'
    )

    # left — conditions
    html_parts.append('<div style="flex:1;min-width:0">')
    html_parts.append('<p class="rcard-sec">Conditions</p>')
    for cond in conds:
        # escape angle brackets so Streamlit won't strip them
        safe = cond.replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(
            f'<div class="rcard-cond">'
            f'<span class="rcard-cond-arrow">›</span>{safe}'
            f'</div>'
        )
    html_parts.append('</div>')

    # right — action pill
    html_parts.append(
        '<div style="flex-shrink:0;text-align:right">'
        '<p class="rcard-sec" style="text-align:right">Action</p>'
        f'<span class="rcard-action" style="color:{ac["color"]};'
        f'background:{ac["bg"]};border:1px solid {ac["color"]}30">'
        f'{ac["icon"]}&nbsp;{ac["label"]}</span>'
        '</div>'
    )

    html_parts.append('</div>')  # end two-col

    # reason
    if rationale:
        safe_r = rationale.replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(
            '<div class="rcard-reason">'
            '<span class="rcard-reason-lbl">Reason:</span>'
            f'<span class="rcard-reason-text">{safe_r}</span>'
            '</div>'
        )

    # metrics footer
    conf_int = int(round(conf))
    conf_bar = max(0, min(100, conf_int))
    html_parts.append('<div class="rcard-footer">')

    # confidence
    html_parts.append(
        '<div class="rcard-metric">'
        '<span class="rcard-mlbl">Confidence</span>'
        f'<span class="rcard-mval">{conf_int}%</span>'
        f'<span class="rcard-track">'
        f'<span class="rcard-fill" style="width:{conf_bar}%"></span></span>'
        '</div>'
    )
    # coverage
    html_parts.append(
        '<div class="rcard-metric">'
        '<span class="rcard-mlbl">Coverage</span>'
        f'<span class="rcard-mval">{cov:.1f}%</span>'
        '</div>'
    )
    if rule.get("precision") is not None:
        prec = float(rule["precision"]) * 100
        html_parts.append(
            '<div class="rcard-metric">'
            '<span class="rcard-mlbl">Precision</span>'
            f'<span class="rcard-mval">{prec:.0f}%</span>'
            '</div>'
        )
    if rule.get("lift") is not None:
        html_parts.append(
            '<div class="rcard-metric">'
            '<span class="rcard-mlbl">Lift</span>'
            f'<span class="rcard-mval">{float(rule["lift"]):.1f}×</span>'
            '</div>'
        )
    html_parts.append('</div>')  # end footer

    html_parts.append('</div>')  # end rcard

    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_band_filter(rules: list[dict[str, Any]]) -> str:
    high   = sum(1 for r in rules if str(r.get("risk_band","")).upper()=="HIGH")
    medium = sum(1 for r in rules if str(r.get("risk_band","")).upper()=="MEDIUM")
    low    = sum(1 for r in rules if str(r.get("risk_band","")).upper()=="LOW")

    st.markdown('<p class="dr-filter-lbl">Filter by band</p>', unsafe_allow_html=True)
    return st.radio(
        "Filter rules",
        options=[
            FILTER_ALL,
            f"{FILTER_HIGH} ({high})",
            f"{FILTER_MED} ({medium})",
            f"{FILTER_LOW} ({low})",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="dr_band_filter",
    )


# ── Page entry point ──────────────────────────────────────────────────────────

def render_decision_rules_page(page_header_fn: Callable[..., None]) -> None:
    inject_styles()

    page_header_fn(
        "Decision Rules",
        "Underwriting policy library — transparent rules aligned with portfolio risk drivers.",
        badge="Policy Engine",
    )

    api_base = st.session_state.get("api_base_url", DEFAULT_API_BASE)
    with st.expander("API connection", expanded=False):
        api_base = st.text_input("Backend URL", value=api_base, key="rules_api_url")
        st.session_state["api_base_url"] = api_base

    try:
        catalog = fetch_rules_catalog(api_base)
    except RulesAPIError as exc:
        st.error(str(exc))
        return

    rules     = _merge_session_matches(catalog.get("rules") or [])
    triggered = [r for r in rules if r.get("matched")]

    # meta row
    st.markdown(
        '<div class="dr-meta">'
        '<span><span class="dr-live-dot"></span>Live policy engine</span>'
        '<span>Derived from LightGBM decision paths</span>'
        f'<span>{len(rules)} rules active</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    render_summary_kpis(rules)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("How these rules work", expanded=False):
        st.markdown(
            """
            **What is real vs generated**

            | Element | Source |
            |---|---|
            | **Conditions** | Structured thresholds evaluated live on each applicant (`/decision`) |
            | **Metrics** | Confidence, coverage, precision, and lift from rule-mining on the training portfolio |
            | **Reason** | Auto-generated from each rule's conditions — not hand-written per rule |
            | **Title** | Short policy summary for scanning (curated label) |

            When you run **Risk Prediction**, matching rules are highlighted here as **TRIGGERED**.
            """
        )

    if triggered:
        names = ", ".join(r.get("title", r.get("rule_id","")) for r in triggered)
        st.markdown(
            f'<div class="dr-trig">'
            f'<span style="font-size:1rem;color:#3b82f6;flex-shrink:0">⚡</span>'
            f'<div><b>{len(triggered)} rule(s) triggered on latest assessment</b><br>{names}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    raw_filter = render_band_filter(rules)

    # strip count suffix so FILTER_TO_BAND still works
    filter_key = raw_filter.split(" (")[0]
    filtered = rules
    if filter_key != FILTER_ALL:
        band_key = FILTER_TO_BAND.get(filter_key)
        if band_key:
            filtered = [r for r in rules if str(r.get("risk_band","")).upper() == band_key]

    st.markdown("<br>", unsafe_allow_html=True)

    if not filtered:
        st.markdown(
            '<div class="dr-empty">◌ No rules match the selected filter.</div>',
            unsafe_allow_html=True,
        )
        return

    for rule in filtered:
        render_rule_card(rule)