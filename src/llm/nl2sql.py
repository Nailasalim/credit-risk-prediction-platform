"""
Natural language → SQL via Groq (DeepSeek R1 Distill Llama 70B).
"""

from __future__ import annotations

import logging
import re

from src.llm.groq_client import chat_completion, nl2sql_model
from src.llm.schema import KPI_TABLE, TABLE_NAME, build_schema_prompt
from src.llm.sql_validator import SqlValidationError, validate_sql

logger = logging.getLogger(__name__)

THINKING_BLOCK = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_thinking_tokens(text: str) -> str:
    """Remove DeepSeek reasoning blocks before SQL parsing."""
    cleaned = THINKING_BLOCK.sub("", text)
    cleaned = re.sub(r"</?redacted_thinking>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_sql(raw: str) -> str:
    """Parse SQL from model output (fenced or bare SELECT)."""
    text = strip_thinking_tokens(raw)
    fence_match = SQL_FENCE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    select_match = re.search(
        r"(SELECT\b[\s\S]*)",
        text,
        re.IGNORECASE,
    )
    if select_match:
        return select_match.group(1).strip().rstrip(";")
    return text.strip()


def build_nl2sql_prompt(question: str, schema: str) -> str:
    """Concise, schema-grounded prompt (single user message for DeepSeek)."""
    return f"""You convert portfolio questions into one SQLite SELECT query.

SCHEMA:
{schema}

SUPPORTED PATTERNS:
1. Aggregation — COUNT, AVG, SUM, MIN, MAX on numeric columns
2. Grouping — GROUP BY categorical columns (education, income type, contract type, gender)
3. Filtering — WHERE on any column
4. Ordering — ORDER BY ASC/DESC with LIMIT
5. Conditional — AND/OR, BETWEEN, comparisons on ranges

RULES:
- Output ONLY the SQL. No markdown, no explanation.
- SELECT only. Single statement. No semicolon at end.
- Use table {TABLE_NAME} or {KPI_TABLE}.
- TARGET=1 means default. Age: ABS(DAYS_BIRTH)/365.25.

QUESTION: {question.strip()}

SQL:"""


def generate_sql_from_question(
    question: str,
    *,
    available_columns: list[str] | None = None,
) -> str:
    """
    Call Groq NL→SQL model and return validated SELECT SQL.

    Raises SqlValidationError or RuntimeError on failure.
    """
    question = question.strip()
    if not question:
        raise SqlValidationError("Question is empty.")

    schema = build_schema_prompt(available_columns)
    prompt = build_nl2sql_prompt(question, schema)

    raw = chat_completion(
        model=nl2sql_model(),
        user_prompt=prompt,
        temperature=0.2,
        max_tokens=400,
    )
    logger.debug("Groq NL→SQL raw response length=%d", len(raw))

    sql = extract_sql(raw)
    return validate_sql(sql)
