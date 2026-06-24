"""
SELECT-only SQL guardrails for AI Data Analyst queries.
"""

from __future__ import annotations

import re

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)

DEFAULT_ROW_LIMIT = 500


class SqlValidationError(ValueError):
    """Raised when generated SQL fails safety checks."""


def strip_sql_comments(sql: str) -> str:
    """Remove line comments before validation."""
    lines = []
    for line in sql.splitlines():
        if "--" in line:
            line = line.split("--", 1)[0]
        lines.append(line)
    return "\n".join(lines).strip()


def ensure_row_limit(sql: str, limit: int = DEFAULT_ROW_LIMIT) -> str:
    """Append LIMIT when the model omits it."""
    cleaned = sql.strip().rstrip(";")
    if re.search(r"\bLIMIT\s+\d+\b", cleaned, re.IGNORECASE):
        return cleaned
    return f"{cleaned} LIMIT {limit}"


def validate_sql(sql: str) -> str:
    """
    Validate and normalize a single SELECT statement.

    Returns cleaned SQL (comments stripped, optional LIMIT applied).
    Raises SqlValidationError when the query is unsafe or malformed.
    """
    if not sql or not sql.strip():
        raise SqlValidationError("SQL query is empty.")

    cleaned = strip_sql_comments(sql).strip().rstrip(";")
    if not cleaned:
        raise SqlValidationError("SQL query is empty after removing comments.")

    upper = cleaned.upper()
    if not upper.startswith("SELECT"):
        raise SqlValidationError("Only SELECT queries are allowed.")

    if FORBIDDEN_KEYWORDS.search(cleaned):
        raise SqlValidationError(
            "Query rejected: DROP, DELETE, INSERT, and UPDATE statements are not permitted."
        )

    if ";" in cleaned:
        raise SqlValidationError("Only a single SQL statement is allowed.")

    return ensure_row_limit(cleaned)
