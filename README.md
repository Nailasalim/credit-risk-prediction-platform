# CreditIQ – AI-Powered Credit Risk Underwriting Platform

CreditIQ is an end-to-end credit risk intelligence workspace built on the **Home Credit Default Risk** dataset. It helps analysts and underwriters move from raw application data to scored decisions, explanations, and portfolio insight in one place.

Traditional underwriting is often **slow**, **inconsistent**, and **hard to explain** to business and compliance stakeholders. This platform combines:

- **Machine learning** — default probability and risk bands  
- **Explainable AI** — SHAP-based drivers at portfolio and applicant level  
- **Business rules** — transparent approve / review / decline policies  
- **Portfolio analytics** — executive KPIs and segment views  
- **Natural language data analysis** — ask portfolio questions in plain English  

…to support faster, more explainable credit decision-making.

| | |
|---|---|
| **Dataset** | Home Credit Default Risk |
| **Records** | 307,511 applications |
| **Target** | Loan default prediction (`TARGET`) |
| **Model** | LightGBM |
| **Explainability** | SHAP (global + local) |
| **Frontend** | Streamlit |
| **Backend** | Python + FastAPI |

**Author:** [Nailasalim](https://github.com/Nailasalim)

---

## Features

Only capabilities that are implemented and available in the app today.

### Executive Dashboard

- Portfolio KPIs (applications, default rate, approval rate, high-risk exposure)  
- Portfolio analytics: risk distribution, approval decision mix, predicted default probability bands, risk segment volume  
- Top risk and positive SHAP drivers  
- Recent session assessments from Risk Prediction  

### Data Explorer

- Dataset overview and KPIs  
- Demographics (age, gender, employment tenure)  
- Financial analysis (income, credit, annuity, scatter views)  
- Risk segmentation and highlights  
- Data quality assessment  
- Interactive Plotly charts with sidebar filters  

### Risk Prediction

- Applicant scoring via FastAPI  
- Probability of default and risk score (0–100)  
- Risk band assignment (Low / Medium / High)  
- Approve / Review / Decline recommendation  
- Session assessment history on the dashboard  

### Explainability

- SHAP-based local explanations per applicant  
- Global feature importance (mean \|SHAP\|)  
- Top drivers and contribution chart  
- Concise AI-style narrative summary  

### Decision Rules

- Underwriting rules engine (structured rule catalog)  
- Live rule matching on applicant payloads  
- Human-readable conditions and rule reasons  
- Confidence, coverage, precision, and lift metrics  

### AI Data Analyst

- Natural language → SQL via **Groq** (`llama-3.3-70b-versatile`)  
- Result summarization via **Groq** (`llama-3.3-70b-versatile`)  
- **SELECT-only** SQL validation (rejects DROP, DELETE, INSERT, UPDATE)  
- Thinking-token stripping for DeepSeek output (`<think>` blocks)  
- Query results as tables + one-line business insights  
- In-memory **SQLite** over `application_train` and portfolio KPIs  

### Login & navigation

- Session-based login (demo accounts)  
- Dark enterprise UI with sidebar navigation across all modules  

---

## Architecture

<p align="center">
  <img src="documents/screenshots/architecture_diag.png" alt="CreditIQ system architecture" width="560" />
</p>

- **Streamlit** — single app shell (`ui/streamlit_app.py`), page modules per feature.  
- **FastAPI** — inference and rules for individual applicants (`/predict`, `/decision`, `/rules`, `/health`, `/dashboard/summary`).  
- **Portfolio batch scoring** — training CSV scored once; metrics cached in `models/portfolio_scoring_snapshot.json` for the executive dashboard.  
- **AI Data Analyst** — loads `application_train` into **SQLite** in-process; Groq converts NL questions to validated SELECT queries and summarizes results.

---

## Machine learning pipeline

| Stage | Description |
|-------|-------------|
| Dataset | Home Credit `application_train` (307,511 labeled rows) |
| Feature engineering | 21 model features (bureau scores, amounts, ratios, tenure, demographics) |
| Imputation | Median imputation (`SimpleImputer`, training-fit) |
| Training | LightGBM classifier |
| Scoring | Default probability at tuned threshold |
| Risk bands | Low / Medium / High from probability cutoffs |
| Decision | Model decision + rules → Approve / Review / Decline |

### Holdout metrics (Phase 1)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7516 |
| Accuracy | 0.8620 |
| Precision | 0.2566 |
| Recall | 0.3742 |
| F1 score | 0.3045 |
| Decision threshold | 0.67 |

---

## Model selection rationale

**LightGBM** was selected as the production classifier for this credit-risk use case because it offers the best balance between **predictive performance** and **interpretability** in a tabular underwriting workflow.

| Consideration | Why LightGBM fits |
|---------------|-------------------|
| Tabular performance | Strong results on structured financial and bureau features |
| Training & inference | Efficient on large applicant volumes (307k+ rows) |
| Heterogeneous inputs | Handles mixed numeric ratios, tenure, and score features natively |
| Feature interactions | Gradient boosting captures non-linear risk combinations |
| Credit-risk fit | Widely used for default / delinquency modelling |
| Explainability | Pairs with SHAP and rule layers for analyst-facing transparency |

LightGBM outperformed simpler baselines in holdout evaluation while remaining practical to deploy behind FastAPI and batch portfolio scoring.

---

## Class imbalance strategy

The Home Credit training set reflects a **real-world imbalanced** outcome distribution:

| Class | Share |
|-------|-------|
| Non-default | 91.9% |
| Default | 8.1% |

**Approach**

- The **natural class distribution was preserved** for training and evaluation — no aggressive oversampling (e.g. SMOTE) was applied in the final deployment pipeline.  
- **Accuracy alone was not relied upon** as the primary success metric given the imbalance.  
- Model quality was assessed with **ROC-AUC**, **precision**, **recall**, and **F1** on holdout data (see table above).  
- **Threshold tuning** (0.67) was used to align predicted default probability with business objectives — balancing approval volume, review workload, and detection of high-risk applicants.

This strategy keeps evaluation honest on rare-default detection while supporting underwriting policy via risk bands and decisions.

---

## Rule derivation logic

Underwriting rules in CreditIQ complement the LightGBM score. They are derived from:

- **Predicted probability of default** and tuned threshold behaviour  
- **Risk band thresholds** (Low / Medium / High)  
- **Portfolio risk policies** aligned with batch-scored segment behaviour  
- **Financial risk indicators** surfaced in the rule catalog (bureau scores, exposure, annuity stress, tenure, etc.)

**Policy flow (band → action)**

| Risk band | Typical action |
|-----------|----------------|
| Low risk | Approve |
| Medium risk | Review |
| High risk | Decline |

Rules do not replace the model; they **translate model output into business language**, highlight which policies fire for an applicant, and improve **transparency** for credit analysts and reviewers. Final recommendations combine model probability, band assignment, and rule evaluation in the Decision Rules and Risk Prediction modules.

---

## Example decision outputs

Representative **system outputs** (illustrative bands and scores — not tied to a specific applicant form submission):

**Prediction output — low risk**

| Field | Value |
|-------|-------|
| Risk score | 22 |
| Risk band | Low |
| Decision | Approve |

**Reason:** Low predicted default probability and acceptable risk profile.

---

**Prediction output — medium risk**

| Field | Value |
|-------|-------|
| Risk score | 54 |
| Risk band | Medium |
| Decision | Review |

**Reason:** Moderate risk indicators require manual assessment.

---

**Prediction output — high risk**

| Field | Value |
|-------|-------|
| Risk score | 81 |
| Risk band | High |
| Decision | Decline |

**Reason:** High predicted default probability and elevated portfolio risk contribution.

---

## Prompt engineering & query design

The **AI Data Analyst** uses the **Groq API** for natural-language portfolio Q&A.

| Step | Model | Role |
|------|-------|------|
| NL → SQL | `llama-3.3-70b-versatile` | Converts plain-English questions into SQLite `SELECT` queries |
| Summarization | `llama-3.3-70b-versatile` | One-line business summary of each result set |

> **Note:** Groq decommissioned `deepseek-r1-distill-llama-70b` (Oct 2025). CreditIQ uses `llama-3.3-70b-versatile` for NL→SQL per [Groq’s deprecation guidance](https://console.groq.com/docs/deprecations). Thinking-token stripping remains for compatibility if a reasoning model is configured via `GROQ_NL2SQL_MODEL`.

**How it works**

1. The user asks a question in plain English (or selects a suggested chip).  
2. A **schema-grounded prompt** (table/column hints for `application_train` and `portfolio_kpis`) is sent to the NL→SQL model.  
3. **Thinking tokens** (`<think>…</think>`) are stripped from DeepSeek output before SQL parsing.  
4. SQL is validated — **SELECT only**; `DROP`, `DELETE`, `INSERT`, and `UPDATE` are rejected.  
5. The query runs against in-memory SQLite.  
6. A compact JSON preview of results is sent to the summarization model for a **one-line business insight**.

**Supported query patterns**

| Pattern | Examples |
|---------|----------|
| Aggregation | `COUNT`, `AVG`, `SUM` over numeric columns |
| Grouping | `GROUP BY` categorical columns (education type, income type, contract type) |
| Filtering | `WHERE` clause conditions on any column |
| Ordering | `ORDER BY` with `ASC` / `DESC` and `LIMIT` |
| Conditional comparisons | Range filters, `AND` / `OR`, `BETWEEN` |

---

## Token optimization strategy

- **Structured prompts** with schema grounding — only curated column hints and KPI definitions are included, not the full 122-column dump.  
- **SELECT-only enforcement** — no open-ended DDL/DML generation.  
- **Thinking-token stripping** — reasoning blocks are removed before SQL extraction to avoid polluting the parser.  
- **Concise summarization prompt** — at most 15 preview rows serialized as compact JSON; output capped to one sentence.  
- **Single user message** for DeepSeek (per Groq guidance) — instructions and question in one prompt.  
- **Automatic `LIMIT 500`** when the model omits a row cap.

---

## Known limitations

- Only **`application_train.csv`** is loaded; bureau, installment, and credit card balance tables are **not** joined.  
- **`EXT_SOURCE_1` / `2` / `3`** are fully anonymised in Home Credit — their real-world meaning is unknown.  
- **Login** uses demo session credentials — not enterprise identity management or per-query API auth.  
- **Groq model availability** — `deepseek-r1-distill-llama-70b` was decommissioned; defaults use `llama-3.3-70b-versatile`. Override via `GROQ_NL2SQL_MODEL` in `.env` if needed.  
- Pre-trained **`models/`** artifacts are mounted in Docker; the ML model is **not** retrained on `docker compose up` (train locally and commit or mount artifacts).

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | API key for Groq inference — obtain from [console.groq.com](https://console.groq.com) |
| `GROQ_NL2SQL_MODEL` | Optional override (default: `llama-3.3-70b-versatile`) |
| `GROQ_SUMMARY_MODEL` | Optional override (default: `llama-3.3-70b-versatile`) |
| `CREDIT_RISK_API_URL` | FastAPI base URL for Risk Prediction / Decision Rules |
| `CREDIT_RISK_PORTFOLIO_CSV` | Path to `application_train.csv` |

Copy [`.env.example`](.env.example) to `.env` and set `GROQ_API_KEY` before using the AI Data Analyst.

> **Note:** `data/`, `models/`, and `.env` are excluded from version control. Place `application_train.csv` in `data/` before running the application.

---

## Portfolio metrics (scored training book)

Batch-scored portfolio used for dashboard and analyst context:

| Metric | Value |
|--------|-------|
| Applications | 307,511 |
| Observed default rate | 8.1% |
| Approval rate (policy) | 56.2% |
| High-risk exposure (HIGH band) | 12.0% |

---

## Screens

All UI captures live in [`documents/screenshots/`](documents/screenshots/). Multiple images per page show scroll / section views.

### Executive Dashboard

Portfolio KPIs, portfolio analytics, and underwriting insights.

| | |
|:---:|:---:|
| Overview & KPIs | Portfolio analytics |
| ![Dashboard 1](documents/screenshots/dashboard1.png) | ![Dashboard 2](documents/screenshots/dashboard2.png) |

### Data Explorer

Demographics, financial analysis, and risk sections.

| | |
|:---:|:---:|
| Overview | Demographics / financial |
| ![Data Explorer 1](documents/screenshots/data_explorer1.png) | ![Data Explorer 2](documents/screenshots/data_explorer2.png) |

![Data Explorer 3 — risk & data quality](documents/screenshots/data_explorer3.png)

### Risk Prediction

Applicant form and scoring result.

| | |
|:---:|:---:|
| Risk Analysis- 1 | Risk Analysis - 2 |
| ![Risk Prediction 1](documents/screenshots/risk_prediction1.png) | ![Risk Prediction 2](documents/screenshots/risk_prediction2.png) |

### Explainability

Applicant-level drivers and SHAP contribution view.

| | |
|:---:|:---:|
| Risk summary & drivers | Custom explainability |
| ![Explainability 1](documents/screenshots/explainability1.png) | ![Explainability 2](documents/screenshots/explainability2.png) |

### Decision Rules

Policy catalog, metrics, and rule detail.

| | |
|:---:|:---:|
| Rules library | Rule evaluation |
| ![Decision Rules 1](documents/screenshots/decision_rules1.png) | ![Decision Rules 2](documents/screenshots/decision_rules2.png) |

![Decision Rules 3](documents/screenshots/decision_rules3.png)

### AI Data Analyst

Natural language query and portfolio insight.

| | |
|:---:|:---:|
| Query & SQL | Results & insight |
| ![AI Data Analyst 1](documents/screenshots/ai_data_analyst1.png) | ![AI Data Analyst 2](documents/screenshots/ai_data_analyst2.png) |

### Model & EDA (training phase)

| Asset | File |
|-------|------|
| ROC curve | `documents/screenshots/roc_curve.png` |
| SHAP summary (training) | `documents/screenshots/shap_summary.png` |
| Confusion matrix | `documents/screenshots/confusion_matrix.png` |

---

## Installation

**Requirements:** Python 3.11+, Git  

**1. Clone and enter the project**

```powershell
git clone https://github.com/Nailasalim/<repo-name>.git
cd credit_risk_prediction
```

**2. Create a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```powershell
pip install -r requirements.txt
```

Use **`scikit-learn==1.7.0`** as pinned in `requirements.txt` (matches `models/imputer.pkl`).

**4. Add the dataset**

Place `application_train.csv` in `data/` (not committed to Git).  
Optional: set `CREDIT_RISK_PORTFOLIO_CSV` to another path.

**5. Configure Groq (AI Data Analyst)**

```powershell
copy .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_from_console.groq.com
```

**6. Set Python path**

```powershell
$env:PYTHONPATH="."
```

---

## Run locally

### Start the API (Risk Prediction, Decision Rules, API-backed flows)

```powershell
$env:PYTHONPATH="."
uvicorn src.api.main:app --reload --port 8000
```

API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Start the Streamlit UI

In a second terminal:

```powershell
$env:PYTHONPATH="."
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | **Required** for AI Data Analyst (Groq NL→SQL + summarization) |
| `GROQ_NL2SQL_MODEL` | `llama-3.3-70b-versatile` | NL → SQL model |
| `GROQ_SUMMARY_MODEL` | `llama-3.3-70b-versatile` | Result summarization model |
| `CREDIT_RISK_API_URL` | `http://127.0.0.1:8000` | UI → FastAPI |
| `CREDIT_RISK_PORTFOLIO_CSV` | `data/application_train.csv` | Dashboard, EDA, AI analyst |

First dashboard load may take ~10–15 seconds while the portfolio snapshot is built or refreshed.

---

## Run with Docker

Docker deployment has been tested and verified locally using the current repository configuration (`Dockerfile`, `docker-compose.yml`).

**Prerequisites:** Docker Desktop, `data/application_train.csv`, `models/` artifacts on the host, and `.env` with `GROQ_API_KEY` for the AI Data Analyst.

```powershell
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend (Streamlit) | [http://localhost:8501](http://localhost:8501) |
| Backend (FastAPI) | [http://localhost:8000](http://localhost:8000) |

The frontend waits for the backend health check and uses `CREDIT_RISK_API_URL=http://backend:8000` inside the Compose network.

> **Note:** Docker configuration is included in the project and has been validated locally. End-to-end execution depends on a working local Docker Desktop environment.

## Docker validation

Local validation checks completed:

- **Build command:** `docker compose up --build`
- **Frontend URL:** `http://localhost:8501`
- **API URL:** `http://localhost:8000`
- **API Docs URL:** `http://localhost:8000/docs`

Validation confirmed successful image build, healthy backend startup, frontend availability, and working Compose networking between `creditiq-frontend` and `creditiq-backend`.

---

## Demo credentials

Primary demo account:

| Field | Value |
|-------|-------|
| **Username** | `analyst` |
| **Password** | `CreditIQ2024` |

Additional demo users (login expander): `admin` / `admin123`, `risk_officer` / `risk2024`.

After sign-in you are redirected to the **Executive Dashboard**.

### Suggested 3-minute demo path

1. **Dashboard** — portfolio KPIs and analytics  
2. **Risk Prediction** — score one applicant (API running)  
3. **Explainability** — review drivers for that applicant  
4. **Decision Rules** — show matched policies  
5. **AI Data Analyst** — e.g. “Summarize portfolio” or a suggested chip  

---

## Project structure

```
credit_risk_prediction/
├── src/api/              # FastAPI
├── src/data/             # Loading, preprocessing, portfolio CSV
├── src/ml/               # Predict, rules, portfolio analytics, SHAP importance
├── ui/                   # Streamlit pages (dashboard, EDA, risk, XAI, rules, analyst, login)
├── models/               # model.pkl, imputer.pkl, metrics, SHAP, portfolio snapshot
├── data/                 # application_train.csv (local)
├── scripts/              # Utility scripts (e.g. imputer fit)
├── documents/            # project_journal.md, findings, screenshots
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Project status

**Current status:** Feature complete for demonstration and portfolio use.

| Module | Status |
|--------|--------|
| Executive Dashboard | ✓ Complete |
| Data Explorer | ✓ Complete |
| Risk Prediction | ✓ Complete |
| Explainability | ✓ Complete |
| Decision Rules | ✓ Complete |
| AI Data Analyst | ✓ Complete |
| Login & app shell | ✓ Complete |
| Docker deployment | ✓ Complete (validated locally) |

Further detail: [documents/project_journal.md](documents/project_journal.md)

---

## Future enhancements

Optional improvements beyond the current demo scope:

- Model monitoring and alert thresholds  
- Feature and prediction drift detection  
- Production authentication (SSO / managed users)  
- Automated retraining and model promotion pipeline  

---

## License

Private submission / portfolio use — all rights reserved unless otherwise specified by the assignment.
