#!/bin/bash
# Start FastAPI + Streamlit in one container (Railway / Render / Fly).
# Public traffic hits Streamlit on $PORT; API listens on localhost:8000.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/app}"
export CREDIT_RISK_API_URL="${CREDIT_RISK_API_URL:-http://127.0.0.1:8000}"
export CREDIT_RISK_PORTFOLIO_CSV="${CREDIT_RISK_PORTFOLIO_CSV:-/app/data/application_train.csv}"
# Prefer cached portfolio KPIs on small Railway plans (avoids OOM scoring 307k rows)
export CREDITIQ_USE_SNAPSHOT_ONLY="${CREDITIQ_USE_SNAPSHOT_ONLY:-1}"

PORT="${PORT:-8501}"

echo "Starting FastAPI on 127.0.0.1:8000 ..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

for i in $(seq 1 30); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" 2>/dev/null; then
    echo "API is healthy."
    break
  fi
  sleep 2
done

echo "Starting Streamlit on 0.0.0.0:${PORT} ..."
exec streamlit run ui/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
