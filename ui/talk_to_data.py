"""
AI Data Analyst — Groq-powered NL → SQL over Home Credit application_train.

NL→SQL: llama-3.3-70b-versatile (Groq)
Summarization: llama-3.3-70b-versatile (Groq)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.portfolio_loader import resolve_portfolio_csv  # noqa: E402
from src.llm.groq_client import get_groq_api_key, nl2sql_model, summary_model  # noqa: E402
from src.llm.nl2sql import generate_sql_from_question  # noqa: E402
from src.llm.sql_validator import SqlValidationError  # noqa: E402
from src.llm.summarize import summarize_query_result  # noqa: E402

SNAPSHOT_PATH = PROJECT_ROOT / "models" / "portfolio_scoring_snapshot.json"
TABLE_NAME = "application_train"

SUGGESTED_QUESTIONS: list[tuple[str, str]] = [
    ("Approval Rate", "What is the approval rate?"),
    ("Default Rate", "What is the default rate?"),
    ("Portfolio Summary", "Summarize the portfolio with total applications and default rate"),
    ("Average Income", "What is the average income?"),
    ("Loan Type Risk", "Which loan type has the highest default rate?"),
    ("Risk By Gender", "Show default rate by gender"),
    ("Top Risk Regions", "What are the top 5 highest risk regions by default rate?"),
]


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    sql: str
    title: str


@dataclass
class QueryResult:
    plan: QueryPlan
    frame: pd.DataFrame
    insight: str
    error: str | None = None


@st.cache_data(show_spinner="Loading Home Credit application data…")
def load_data() -> pd.DataFrame:
    path = resolve_portfolio_csv()
    if path is None:
        raise FileNotFoundError(
            "application_train.csv not found. Place it at data/application_train.csv "
            "or set CREDIT_RISK_PORTFOLIO_CSV."
        )
    return pd.read_csv(path)


def _load_portfolio_kpis() -> dict[str, float]:
    """Cached underwriting KPIs from portfolio batch scoring (display / SQL seed)."""
    if not SNAPSHOT_PATH.is_file():
        return {}
    with SNAPSHOT_PATH.open(encoding="utf-8") as file:
        snap = json.load(file)
    return {
        "approval_rate_pct": float(snap.get("approval_rate_pct", 0)),
        "decline_rate_pct": float(snap.get("decline_rate_pct", 0)),
        "review_rate_pct": float(snap.get("review_rate_pct", 0)),
        "observed_default_rate_pct": float(snap.get("observed_default_rate_pct", 0)),
        "scored_records": float(snap.get("scored_records", 0)),
        "threshold": float(snap.get("threshold", 0.67)),
    }


@st.cache_resource(show_spinner="Preparing in-memory analytics database…")
def create_database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    frame = load_data()
    frame.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")

    kpis = _load_portfolio_kpis()
    conn.execute(
        """
        CREATE TABLE portfolio_kpis (
            metric TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
        """
    )
    if kpis:
        rows = [
            ("approval_rate_pct", kpis["approval_rate_pct"]),
            ("decline_rate_pct", kpis["decline_rate_pct"]),
            ("review_rate_pct", kpis["review_rate_pct"]),
            ("observed_default_rate_pct", kpis["observed_default_rate_pct"]),
            ("scored_records", kpis["scored_records"]),
            ("decision_threshold", kpis["threshold"]),
        ]
        conn.executemany("INSERT INTO portfolio_kpis (metric, value) VALUES (?, ?)", rows)
    conn.commit()
    return conn


def _table_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
    return [row[1] for row in rows]


def _format_query_error(exc: Exception) -> str:
    """User-friendly Groq / query errors."""
    message = str(exc)
    if "model_decommissioned" in message or "decommissioned" in message.lower():
        return (
            "The configured Groq NL→SQL model is no longer available. "
            "Set GROQ_NL2SQL_MODEL=llama-3.3-70b-versatile in .env and restart Streamlit."
        )
    if "GROQ_API_KEY" in message:
        return message
    return f"Query failed: {message}"


def run_analyst_query(question: str) -> QueryResult:
    """NL → validated SQL → execute → Groq business summary."""
    question = question.strip()
    if not question:
        plan = QueryPlan("empty", "-- Enter a portfolio question.", "Empty")
        return QueryResult(
            plan=plan,
            frame=pd.DataFrame(),
            insight="Ask a question about the portfolio using the chat box or suggested chips.",
        )

    try:
        get_groq_api_key()
    except RuntimeError as exc:
        plan = QueryPlan("config_error", "-- GROQ_API_KEY not configured", "Configuration")
        return QueryResult(plan=plan, frame=pd.DataFrame(), insight=str(exc), error=str(exc))

    plan = QueryPlan("pending", "-- Generating SQL via Groq…", question[:72])

    try:
        conn = create_database()
        columns = _table_columns(conn)
        sql = generate_sql_from_question(question, available_columns=columns)
        plan = QueryPlan("groq_nl2sql", sql, question[:72])
        frame = pd.read_sql_query(sql, conn)
        insight = summarize_query_result(question, sql, frame)
        return QueryResult(plan=plan, frame=frame, insight=insight)
    except SqlValidationError as exc:
        return QueryResult(
            plan=plan,
            frame=pd.DataFrame(),
            insight="The generated SQL failed safety validation. Try rephrasing your question.",
            error=str(exc),
        )
    except Exception as exc:
        friendly = _format_query_error(exc)
        return QueryResult(
            plan=plan,
            frame=pd.DataFrame(),
            insight=friendly,
            error=friendly,
        )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def inject_talk_to_data_styles() -> None:
    st.markdown(
        """
        <style>
        .tda-hero {
            margin-bottom: 1rem; padding-bottom: 0.85rem;
            border-bottom: 1px solid #243044;
        }
        .tda-eyebrow {
            font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.12em; color: #3b82f6; margin: 0 0 0.35rem 0;
        }
        .tda-title {
            font-size: 1.75rem; font-weight: 700; color: #f1f5f9; margin: 0;
            letter-spacing: -0.03em;
        }
        .tda-sub {
            font-size: 0.88rem; color: #8b9bb4; margin: 0.4rem 0 0 0; line-height: 1.5;
        }
        .tda-panel-label {
            font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.08em; color: #6b7c96; margin: 0 0 0.5rem 0;
        }
        .tda-bubble-user {
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-radius: 10px 10px 10px 2px;
            padding: 0.65rem 0.85rem; margin: 0.5rem 0;
            font-size: 0.84rem; color: #e8edf5;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button {
            font-size: 0.72rem !important;
            padding: 0.25rem 0.55rem !important;
            border-radius: 999px !important;
            background: #121820 !important;
            border: 1px solid #243044 !important;
            color: #94a3b8 !important;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button:hover {
            border-color: #3b82f6 !important;
            color: #e8edf5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_chat_state() -> None:
    if "tda_messages" not in st.session_state:
        st.session_state.tda_messages = []
    if "tda_active_result" not in st.session_state:
        st.session_state.tda_active_result = None


def _process_question(question: str) -> None:
    question = question.strip()
    if not question:
        return
    st.session_state.tda_messages.append({"role": "user", "content": question})
    with st.spinner("Generating SQL and insight via Groq…"):
        result = run_analyst_query(question)
    st.session_state.tda_active_result = result
    st.session_state.tda_messages.append(
        {
            "role": "assistant",
            "content": result.insight,
            "sql": result.plan.sql,
            "title": result.plan.title,
        }
    )


def _render_chat_history() -> None:
    for msg in st.session_state.tda_messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="tda-bubble-user"><strong>You</strong><br/>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.container(border=True):
                st.caption("Analyst")
                st.markdown(msg["content"])


def _render_suggested_chips() -> None:
    st.markdown('<p class="tda-panel-label">Suggested questions</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (label, question) in enumerate(SUGGESTED_QUESTIONS):
        with cols[idx % 3]:
            if st.button(label, key=f"tda_chip_{idx}", use_container_width=True):
                _process_question(question)
                st.rerun()


def _render_analytics_panel(result: QueryResult | None) -> None:
    st.markdown('<p class="tda-panel-label">Query workspace</p>', unsafe_allow_html=True)

    if result is None:
        st.info("Ask a question or select a chip to generate SQL and portfolio analytics.")
        return

    if result.error:
        st.error(result.error)

    st.markdown("**Generated SQL**")
    st.code(result.plan.sql, language="sql")

    st.markdown("**Results**")
    if result.frame.empty:
        st.caption("No rows returned.")
    else:
        st.dataframe(
            result.frame,
            use_container_width=True,
            hide_index=True,
            height=min(320, 48 + 35 * len(result.frame)),
        )

    st.markdown(
        '<p class="tda-panel-label" style="margin-top:0.75rem">Business insight</p>',
        unsafe_allow_html=True,
    )
    st.markdown(result.insight)


def render_talk_to_data_page(page_header_fn: Callable[..., None]) -> None:
    del page_header_fn
    inject_talk_to_data_styles()
    _init_chat_state()

    st.markdown(
        """
        <div class="tda-hero">
            <p class="tda-eyebrow">Enterprise Risk Platform</p>
            <h1 class="tda-title">AI Data Analyst</h1>
            <p class="tda-sub">
                Ask questions in plain English — Groq converts them to validated SQL and
                summarizes results for underwriting and portfolio review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        create_database()
        get_groq_api_key()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except RuntimeError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Failed to initialize analytics database: {exc}")
        return

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<p class="tda-panel-label">Conversation</p>', unsafe_allow_html=True)
        _render_suggested_chips()

        chat_box = st.container(height=380, border=False)
        with chat_box:
            if not st.session_state.tda_messages:
                st.caption("Start with a suggested question or type your own below.")
            else:
                _render_chat_history()

        prompt = st.chat_input("Ask about defaults, approval rate, income, segments…")
        if prompt:
            _process_question(prompt)
            st.rerun()

        if st.button("Clear conversation", type="secondary"):
            st.session_state.tda_messages = []
            st.session_state.tda_active_result = None
            st.rerun()

    with right:
        _render_analytics_panel(st.session_state.tda_active_result)

    st.caption(
        f"Groq NL→SQL · {nl2sql_model()} · Summarization · {summary_model()} · "
        "SELECT-only validation · in-memory SQLite · `application_train`"
    )
