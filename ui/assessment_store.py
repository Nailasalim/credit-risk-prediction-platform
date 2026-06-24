"""
Session-scoped underwriting assessments (Risk Prediction → Dashboard / Explainability).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

HISTORY_KEY = "assessment_history"
SEEN_KEYS = "assessment_seen_fingerprints"
LATEST_KEY = "latest_assessment"
MAX_HISTORY = 20


def _fingerprint(
    applicant_name: str,
    applicant_id: str,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> str:
    blob = json.dumps(
        {
            "applicant_name": applicant_name,
            "applicant_id": applicant_id,
            "payload": payload,
            "risk_score": result.get("risk_score"),
            "default_probability": result.get("default_probability"),
            "risk_band": result.get("risk_band"),
            "recommendation": result.get("recommendation"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def record_assessment(
    applicant_name: str,
    applicant_id: str,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Persist a completed underwriting assessment in session state."""
    name = (applicant_name or "Unnamed applicant").strip()
    app_id = (applicant_id or "—").strip()
    recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    record: dict[str, Any] = {
        "fingerprint": _fingerprint(name, app_id, payload, result),
        "applicant_name": name,
        "applicant_id": app_id,
        "recorded_at": recorded_at,
        "timestamp": recorded_at,
        "risk_score": int(result.get("risk_score", 0)),
        "risk_band": str(result.get("risk_band", "")).upper(),
        "default_probability": float(result.get("default_probability", 0)),
        "default_probability_pct": float(result.get("default_probability", 0)),
        "recommendation": str(result.get("recommendation", "")).upper(),
        "amt_credit": float(payload.get("AMT_CREDIT", 0)),
        "ext_source_2": float(payload.get("EXT_SOURCE_2", 0)),
    }

    st.session_state[LATEST_KEY] = record
    st.session_state["last_applicant_payload"] = dict(payload)
    st.session_state["risk_result"] = dict(result)

    seen: set[str] = st.session_state.setdefault(SEEN_KEYS, set())
    if record["fingerprint"] not in seen:
        seen.add(record["fingerprint"])
        history: list[dict[str, Any]] = st.session_state.setdefault(HISTORY_KEY, [])
        history.insert(0, record)
        st.session_state[HISTORY_KEY] = history[:MAX_HISTORY]

    return record


def get_latest_assessment() -> dict[str, Any] | None:
    latest = st.session_state.get(LATEST_KEY)
    if latest:
        return dict(latest)
    result = st.session_state.get("risk_result")
    payload = st.session_state.get("last_applicant_payload")
    if result and payload:
        return {
            "applicant_name": st.session_state.get("applicant_name", "Latest applicant"),
            "applicant_id": st.session_state.get("applicant_id", "—"),
            "recorded_at": "",
            "timestamp": "",
            "risk_score": int(result.get("risk_score", 0)),
            "risk_band": str(result.get("risk_band", "")).upper(),
            "default_probability": float(result.get("default_probability", 0)),
            "recommendation": str(result.get("recommendation", "")).upper(),
        }
    return None


def list_assessments() -> list[dict[str, Any]]:
    return list(st.session_state.get(HISTORY_KEY, []))
