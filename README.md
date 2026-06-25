# CreditIQ – AI-Powered Credit Risk Underwriting Platform

End-to-end credit risk workspace on the **Home Credit Default Risk** dataset (307,511 applications). Combines **LightGBM scoring**, **SHAP explainability**, **business rules**, **portfolio analytics**, and a **Groq-powered AI Data Analyst**.

| | |
|---|---|
| **Model** | LightGBM · 109 features · `scale_pos_weight` |
| **Holdout** | ROC-AUC **0.762** · PR-AUC **0.252** · Recall **0.434** |
| **LLM** | Groq NL→SQL + summarization (`llama-3.3-70b-versatile`) |
| **Stack** | Streamlit · FastAPI · SQLite · Docker |

**Author:** [Nailasalim](https://github.com/Nailasalim)

<p align="center">
  <img src="documents/screenshots/architecture_diag.png" alt="CreditIQ architecture" width="560" />
</p>

---

## Highlights

- **Executive Dashboard** — portfolio KPIs, risk bands, SHAP drivers, batch-scored analytics
- **Risk Prediction** — real-time default probability, risk score, Approve / Review / Decline
- **Explainability** — per-applicant SHAP contributions (red / green bar chart)
- **Decision Rules** — 7 policy rules with live matching and business-readable conditions
- **AI Data Analyst** — plain English → validated SQL → Groq one-line business insight
- **Data Explorer** — interactive EDA with Plotly filters
- **Docker** — validated local deployment (`docker compose up --build`)

---

## Machine learning

| Stage | Detail |
|-------|--------|
| Features | **109** — label-encoded categoricals, financial ratios, `application_train` only |
| Imbalance | **`scale_pos_weight` ≈ 11.4** — no SMOTE |
| Imputation | Median `SimpleImputer` (train-split fit) |
| Threshold | **0.65** (F1-tuned on holdout) |

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.762 |
| PR-AUC | 0.252 |
| Recall | 0.434 |
| F1 | 0.316 |
| Accuracy | 0.848 |

**Design choices:** LightGBM for tabular speed and native class weighting; SHAP for transparency; rules layer for auditability. Top drivers: `EXT_SOURCE_1/2/3`, tenure, financial ratios.

**Retrain:**

```powershell
python scripts/train_model.py
python scripts/build_portfolio_snapshot.py
```

---

## AI Data Analyst (Groq)

| Step | Implementation |
|------|----------------|
| NL → SQL | Groq + schema-grounded prompt → SQLite `SELECT` |
| Safety | Rejects `DROP` / `DELETE` / `INSERT` / `UPDATE`; auto `LIMIT 500` |
| Summary | Second Groq call → one-line business insight |

Copy `.env.example` → `.env` and set `GROQ_API_KEY` from [console.groq.com](https://console.groq.com).

---

## Quick start

**Prerequisites:** Python 3.11+, `data/application_train.csv`, `models/` artifacts

```powershell
git clone https://github.com/Nailasalim/<your-repo>.git
cd credit_risk_prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # add GROQ_API_KEY
```

**Terminal 1 — API**

```powershell
$env:PYTHONPATH="."
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 — UI**

```powershell
$env:PYTHONPATH="."
streamlit run ui/streamlit_app.py
```

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| API / docs | http://localhost:8000/docs |

**Docker**

```powershell
docker compose up --build
```

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Required for AI Data Analyst |
| `CREDIT_RISK_API_URL` | UI → FastAPI (default `http://127.0.0.1:8000`) |
| `CREDIT_RISK_PORTFOLIO_CSV` | Path to `application_train.csv` |

> `.env` and `data/*.csv` are not committed. Place the dataset in `data/` locally.

---

## Demo login

| Username | Password |
|----------|----------|
| `analyst` | `CreditIQ2024` |

Also: `admin` / `admin123`, `risk_officer` / `risk2024`

**3-min demo:** Dashboard → Risk Prediction → Explainability → Decision Rules → AI Data Analyst

---

## Screenshots

| Dashboard | Data Explorer | Risk Prediction |
|:---:|:---:|:---:|
| ![Dashboard](documents/screenshots/dashboard1.png) | ![EDA](documents/screenshots/data_explorer1.png) | ![Risk](documents/screenshots/risk_prediction1.png) |

| Explainability | Decision Rules | AI Analyst |
|:---:|:---:|:---:|
| ![XAI](documents/screenshots/explainability1.png) | ![Rules](documents/screenshots/decision_rules1.png) | ![Analyst](documents/screenshots/ai_data_analyst1.png) |

More captures: [`documents/screenshots/`](documents/screenshots/)

---

## Project structure

```
credit_risk_prediction/
├── src/api/           # FastAPI inference
├── src/data/          # Feature engineering (109), loaders
├── src/llm/           # Groq NL→SQL + summarization
├── src/ml/            # Predict, rules, portfolio analytics
├── ui/                # Streamlit pages
├── models/            # model.pkl, encoders, metrics, SHAP, snapshot
├── scripts/           # train_model.py, portfolio snapshot
├── Dockerfile
└── docker-compose.yml
```

---

## Status

| Module | Status |
|--------|--------|
| Dashboard · EDA · Risk · XAI · Rules · AI Analyst · Login · Docker | ✓ Complete |

---

## License

Private portfolio / submission use.
