"""
Cross-page Streamlit navigation via session_state.

Use request_page() + apply_pending_navigation() to change pages before the
sidebar navigation buttons are drawn.
"""

from __future__ import annotations

import streamlit as st

PENDING_PAGE_KEY = "pending_page"

PAGE_EXPLAINABILITY = "explainability"

EXPLAIN_MODE_LATEST = "Use latest Risk Prediction assessment"
EXPLAIN_MODE_CUSTOM = "Explain custom applicant"


def request_page(page_id: str) -> None:
    """Queue a page change for the next run (before the nav widget is created)."""
    st.session_state[PENDING_PAGE_KEY] = page_id


def apply_pending_navigation(valid_page_ids: list[str]) -> None:
    """
    Apply a queued navigation request by setting main_nav.

    Call this at the start of render_sidebar(), before nav buttons render.
    """
    pending = st.session_state.pop(PENDING_PAGE_KEY, None)
    if pending and pending in valid_page_ids:
        st.session_state.main_nav = pending


def navigate_to_explainability_latest() -> None:
    """Open Explainability and load the most recent Risk Prediction result."""
    request_page(PAGE_EXPLAINABILITY)
    st.session_state.explainability_mode = EXPLAIN_MODE_LATEST
    st.session_state.explainability_open_latest = True
