# CreditIQ — shared image for FastAPI backend and Streamlit frontend
FROM python:3.11-slim

WORKDIR /app

# LightGBM runtime dependency
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hosted single-process entry (Railway / Render); local compose overrides CMD
RUN chmod +x scripts/start_hosted.sh

EXPOSE 8000
EXPOSE 8501

# Local default = API only. Hosted platforms set start command to:
#   bash scripts/start_hosted.sh
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
