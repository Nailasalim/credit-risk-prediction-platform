"""
One-line business summaries of query results via Groq (Llama 3.3 70B).
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from src.llm.groq_client import chat_completion, summary_model

logger = logging.getLogger(__name__)

MAX_PREVIEW_ROWS = 15


def _compact_result_payload(frame: pd.DataFrame) -> str:
    """Serialize a small result preview for the summarization prompt."""
    preview = frame.head(MAX_PREVIEW_ROWS)
    payload = {
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "preview_rows": preview.to_dict(orient="records"),
    }
    return json.dumps(payload, default=str)


def build_summary_prompt(question: str, sql: str, frame: pd.DataFrame) -> str:
    return f"""Summarize this credit portfolio query result in exactly ONE concise business sentence.
No bullet points. No SQL. Plain English for a risk analyst.

Question: {question.strip()}
SQL: {sql.strip()}
Results: {_compact_result_payload(frame)}

One-line summary:"""


def summarize_query_result(question: str, sql: str, frame: pd.DataFrame) -> str:
    """Return a one-line Groq-generated business insight."""
    if frame.empty:
        return "No rows matched the query filters; refine the question or check column values."

    prompt = build_summary_prompt(question, sql, frame)
    summary = chat_completion(
        model=summary_model(),
        user_prompt=prompt,
        temperature=0.3,
        max_tokens=120,
    )
    line = " ".join(summary.strip().splitlines())
    if len(line) > 320:
        line = line[:317].rstrip() + "..."
    logger.info("Groq summary generated (%d chars)", len(line))
    return line
