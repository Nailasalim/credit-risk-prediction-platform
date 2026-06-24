"""
Thin Groq API client for AI Data Analyst (NL→SQL + summarization).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from groq import Groq

from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_NL2SQL_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SUMMARY_MODEL = "llama-3.3-70b-versatile"


def _load_dotenv() -> None:
    """Load .env from project root (Streamlit does not load this automatically)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ[key] = value


def get_groq_api_key() -> str:
    _load_dotenv()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env in the project root, "
            "add your key from console.groq.com, then restart Streamlit."
        )
    return key


def nl2sql_model() -> str:
    _load_dotenv()
    return os.environ.get("GROQ_NL2SQL_MODEL", DEFAULT_NL2SQL_MODEL).strip()


def summary_model() -> str:
    _load_dotenv()
    return os.environ.get("GROQ_SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL).strip()


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    return Groq(api_key=get_groq_api_key())


def chat_completion(
    *,
    model: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Single-turn Groq chat completion; returns assistant text."""
    client = get_groq_client()
    logger.info("Groq request | model=%s tokens_max=%d", model, max_tokens)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Empty response from Groq model {model}")
    return content.strip()
