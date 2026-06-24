"""
Credit Risk Intelligence Platform — Streamlit UI shell.

Enterprise dashboard layout with sidebar navigation. Backend integration
is intentionally deferred; pages render placeholders until wired to the API.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

APP_TITLE = "CreditIQ"
APP_SUBTITLE = "Enterprise Risk Platform"
APP_EYEBROW = "Enterprise Risk Platform"
APP_VERSION = "v0.1.0-shell"

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

NAV_ITEMS: list[tuple[str, str, str]] = [
    ("dashboard", "Dashboard", "📊"),
    ("data_explorer", "Data Explorer", "🗂️"),
    ("risk_prediction", "Risk Prediction", "🎯"),
    ("explainability", "Explainability", "🔍"),
    ("decision_rules", "Decision Rules", "⚖️"),
    ("talk_to_data", "AI Data Analyst", "💬"),
]

NAV_GROUPS: list[tuple[str | None, list[str]]] = [
    (None, ["dashboard", "data_explorer"]),
    ("Credit Decisions", ["risk_prediction", "explainability", "decision_rules"]),
    ("Intelligence", ["talk_to_data"]),
]

NAV_BY_ID: dict[str, tuple[str, str]] = {
    page_id: (label, icon) for page_id, label, icon in NAV_ITEMS
}

PAGE_RENDERERS: dict[str, Callable[[], None]] = {}


def register_page(page_id: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Decorator to register a page render function."""

    def decorator(func: Callable[[], None]) -> Callable[[], None]:
        PAGE_RENDERERS[page_id] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Theme & global styles
# ---------------------------------------------------------------------------


def inject_theme() -> None:
    """Inject dark banking / risk analytics theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

        :root {
            --bg-app: #0b0f14;
            --bg-panel: #121820;
            --bg-card: #161d27;
            --bg-card-hover: #1a2330;
            --border: #243044;
            --text: #e8edf5;
            --text-muted: #8b9bb4;
            --accent: #3b82f6;
            --accent-soft: rgba(59, 130, 246, 0.15);
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --radius: 12px;
        }

        .stApp {
            background: linear-gradient(165deg, #0b0f14 0%, #0e1319 45%, #0b1018 100%);
            font-family: 'DM Sans', sans-serif;
        }

        /* Remove Streamlit top chrome gap; keep a small safe inset only */
        header[data-testid="stHeader"],
        [data-testid="stHeader"] {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            overflow: hidden !important;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none !important;
        }

        [data-testid="stAppViewContainer"] {
            top: 0 !important;
        }

        [data-testid="stAppViewContainer"] > section.main,
        section[data-testid="stMain"],
        section.main {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }

        section.main > div.block-container,
        [data-testid="stMainBlockContainer"],
        .main .block-container {
            padding-top: 0.35rem !important;
            padding-bottom: 1.25rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100%;
        }

        /* No extra gap above the first widget / hero on each page */
        section.main [data-testid="stVerticalBlock"]:first-of-type {
            padding-top: 0 !important;
            gap: 0.35rem !important;
        }

        section.main [data-testid="stVerticalBlock"] > div:first-child {
            padding-top: 0 !important;
        }

        section.main .element-container:first-of-type {
            margin-top: 0 !important;
        }

        section.main [data-testid="stMarkdownContainer"]:first-of-type {
            margin-top: 0 !important;
        }

        /* Sidebar — stable width; hide native collapse chrome that breaks custom nav */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1018 0%, #080b10 100%) !important;
            border-right: 1px solid #1a2433;
            width: 16.5rem !important;
            min-width: 16.5rem !important;
            max-width: 16.5rem !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
            padding-top: 0.35rem !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding: 0.35rem 0.75rem 1rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:first-of-type {
            padding-top: 0 !important;
        }

        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        button[kind="header"] {
            display: none !important;
        }

        /* Collapsed st.radio labels sometimes leak widget key names as visible "key" */
        .stRadio > label[data-testid="stWidgetLabel"],
        [data-testid="stRadio"] > label {
            display: none !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }

        /* Page hero blocks — align with sidebar top; minimal safe spacing */
        .dash-hero, .de-hero, .tda-hero, .xai-hero {
            margin-top: 0 !important;
            margin-bottom: 0.6rem !important;
            padding-top: 0 !important;
            padding-bottom: 0.5rem !important;
        }

        h1, h2, h3, h4, p, label, span {
            font-family: 'DM Sans', sans-serif !important;
        }

        /* Hide default Streamlit chrome for cleaner shell */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Sidebar brand */
        .sidebar-brand {
            padding: 0.05rem 0.2rem 0.65rem;
            border-bottom: 1px solid #1e2a3a;
            margin-bottom: 0.55rem;
        }
        .sidebar-logo {
            display: inline-flex; align-items: center; justify-content: center;
            width: 2rem; height: 2rem; border-radius: 8px;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            font-size: 1rem; margin-bottom: 0.55rem;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        }
        .brand-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.2;
        }
        .brand-sub {
            font-size: 0.68rem;
            color: var(--text-muted);
            margin: 0.2rem 0 0 0;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }
        .brand-accent {
            color: var(--accent);
        }
        .nav-section {
            font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.1em; color: #4a5d78;
            margin: 0.85rem 0 0.35rem 0.35rem;
            padding: 0;
        }
        .nav-section:first-of-type { margin-top: 0.05rem; }

        [data-testid="stSidebar"] .stButton {
            margin-bottom: 0.15rem;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            justify-content: flex-start;
            text-align: left;
            font-size: 0.82rem;
            font-weight: 500;
            padding: 0.5rem 0.65rem 0.5rem 0.75rem;
            border-radius: 8px;
            border: 1px solid transparent;
            background: transparent;
            color: #94a3b8;
            box-shadow: none;
            transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(59, 130, 246, 0.08);
            color: #e8edf5;
            border-color: #243044;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
            background: rgba(59, 130, 246, 0.14) !important;
            color: #f1f5f9 !important;
            border-color: rgba(59, 130, 246, 0.35) !important;
            border-left: 3px solid #3b82f6 !important;
            padding-left: 0.6rem !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] .stButton > button:focus:not(:active) {
            border-color: transparent;
            color: #94a3b8;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:focus {
            border-color: rgba(59, 130, 246, 0.35) !important;
            color: #f1f5f9 !important;
        }

        /* Page header */
        .page-header {
            margin-bottom: 1.5rem;
        }
        .page-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.03em;
        }
        .page-desc {
            font-size: 0.95rem;
            color: var(--text-muted);
            margin: 0;
            max-width: 720px;
            line-height: 1.5;
        }
        .page-badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            background: var(--accent-soft);
            color: var(--accent);
            margin-bottom: 0.5rem;
        }

        /* KPI cards */
        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.1rem 1.25rem;
            height: 100%;
            transition: border-color 0.2s, background 0.2s;
        }
        .kpi-card:hover {
            background: var(--bg-card-hover);
            border-color: #2f3f56;
        }
        .kpi-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--text-muted);
            margin: 0 0 0.5rem 0;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text);
            margin: 0;
            letter-spacing: -0.02em;
        }
        .kpi-delta {
            font-size: 0.8rem;
            margin: 0.45rem 0 0 0;
            font-weight: 500;
        }
        .kpi-delta.positive { color: var(--success); }
        .kpi-delta.negative { color: var(--danger); }
        .kpi-delta.neutral { color: var(--text-muted); }
        .kpi-icon {
            float: right;
            font-size: 1.25rem;
            opacity: 0.85;
        }

        /* Section panels */
        .panel-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1rem 1.15rem;
            margin-bottom: 0.5rem;
        }
        .panel-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text);
            margin: 0 0 0.25rem 0;
        }
        .panel-subtitle {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin: 0 0 0.75rem 0;
        }

        /* Placeholder banner */
        .placeholder-banner {
            background: rgba(59, 130, 246, 0.08);
            border: 1px dashed #2a4060;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }

        /* Sidebar footer + signed-in user */
        .nav-footer {
            font-size: 0.62rem;
            color: #5c6b82;
            margin-top: 1.25rem;
            padding: 0.65rem 0.35rem 0;
            border-top: 1px solid #1e2a3a;
            line-height: 1.45;
        }

        .sidebar-user-footer {
            margin-top: 0.85rem;
            padding: 0.7rem 0.75rem;
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        .sidebar-user-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }
        .sidebar-user-avatar {
            flex-shrink: 0;
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 50%;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.95rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
        }
        .sidebar-user-meta {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.12rem;
        }
        .sidebar-user-name {
            display: block;
            font-size: 0.84rem;
            font-weight: 600;
            color: var(--text);
            line-height: 1.25;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .sidebar-user-id {
            display: block;
            font-size: 0.68rem;
            color: var(--text-muted);
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        [data-testid="stSidebar"] button[data-testid="baseButton-auth_sign_out"] {
            margin-top: 0.5rem !important;
            min-height: 2rem !important;
            font-size: 0.78rem !important;
        }
        .nav-footer strong {
            color: #8b9bb4;
            font-weight: 600;
        }
        .nav-model-pill {
            display: inline-block;
            margin-top: 0.35rem;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            background: rgba(34, 197, 94, 0.1);
            color: #4ade80;
            font-size: 0.58rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }

        /* Streamlit widget tuning */
        .stDataFrame { border-radius: var(--radius); overflow: hidden; }
        div[data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.75rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def render_brand_block() -> None:
    """Sidebar brand header."""
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-logo">🏦</div>
            <p class="brand-title"><span class="brand-accent">Credit</span>IQ</p>
            <p class="brand-sub">{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    title: str,
    description: str,
    badge: str | None = None,
) -> None:
    """Consistent page title block."""
    badge_html = f'<span class="page-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="page-header">
            {badge_html}
            <h1 class="page-title">{title}</h1>
            <p class="page-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def placeholder_banner(message: str) -> None:
    """Inform users that a section is not yet wired to the backend."""
    st.markdown(
        f'<div class="placeholder-banner">ℹ️ {message}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    delta: str,
    delta_tone: str = "neutral",
    icon: str = "📈",
) -> None:
    """
    Render a single KPI card.

    delta_tone: 'positive' | 'negative' | 'neutral'
    """
    st.markdown(
        f"""
        <div class="kpi-card">
            <span class="kpi-icon">{icon}</span>
            <p class="kpi-label">{label}</p>
            <p class="kpi-value">{value}</p>
            <p class="kpi-delta {delta_tone}">{delta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict[str, str]], columns: int = 4) -> None:
    """Render a responsive row of KPI cards."""
    cols = st.columns(columns)
    for col, card in zip(cols, cards):
        with col:
            kpi_card(**card)


def panel_section(title: str, subtitle: str = "") -> None:
    """Section title inside a content panel."""
    sub = f'<p class="panel-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="panel-card" style="margin-bottom: 0.75rem;">
            <p class="panel-title">{title}</p>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_shell(
    title: str,
    description: str,
    badge: str | None = None,
    show_placeholder: bool = True,
) -> None:
    """Standard wrapper for non-dashboard pages."""
    page_header(title, description, badge=badge)
    if show_placeholder:
        placeholder_banner(
            "UI shell only — backend integration will connect this page to the API."
        )


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------


@register_page("dashboard")
def render_dashboard() -> None:
    from dashboard_page import render_executive_dashboard

    render_executive_dashboard(page_header)


@register_page("data_explorer")
def render_data_explorer() -> None:
    from data_explorer_page import render_data_explorer_page

    render_data_explorer_page(page_header)


@register_page("risk_prediction")
def render_risk_prediction() -> None:
    from risk_prediction_page import render_risk_prediction_page

    render_risk_prediction_page(page_header)


@register_page("explainability")
def render_explainability() -> None:
    from explainability_page import render_explainability_page

    render_explainability_page(page_header)


@register_page("decision_rules")
def render_decision_rules() -> None:
    from decision_rules_page import render_decision_rules_page

    render_decision_rules_page(page_header)


@register_page("talk_to_data")
def render_talk_to_data() -> None:
    from talk_to_data import render_talk_to_data_page

    render_talk_to_data_page(page_header)


# ---------------------------------------------------------------------------
# Sidebar & routing
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Build sidebar navigation and return selected page id."""
    from app_navigation import apply_pending_navigation

    ids = [page_id for page_id, _, _ in NAV_ITEMS]
    if "main_nav" not in st.session_state:
        st.session_state.main_nav = "dashboard"

    apply_pending_navigation(ids)

    with st.sidebar:
        render_brand_block()

        for group_name, group_ids in NAV_GROUPS:
            if group_name:
                st.markdown(f'<p class="nav-section">{group_name}</p>', unsafe_allow_html=True)
            for page_id in group_ids:
                label, _icon = NAV_BY_ID[page_id]
                is_active = st.session_state.main_nav == page_id
                if st.button(
                    label,
                    key=f"nav_btn_{page_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.main_nav = page_id
                    st.rerun()

        st.markdown(
            f"""
            <div class="nav-footer">
                <strong>Model</strong> LightGBM · AUC 0.75
                <span class="nav-model-pill">PRODUCTION</span><br/>
                {APP_VERSION}
            </div>
            """,
            unsafe_allow_html=True,
        )

        from login_page import render_sidebar_user_block

        render_sidebar_user_block()

        return str(st.session_state.main_nav)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} | {APP_SUBTITLE}",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_theme()

    from login_page import is_authenticated, render_login_page

    if not is_authenticated():
        render_login_page()
        return

    page_id = render_sidebar()

    renderer = PAGE_RENDERERS.get(page_id, render_dashboard)
    renderer()


if __name__ == "__main__":
    main()
