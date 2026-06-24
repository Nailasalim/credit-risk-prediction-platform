"""
Generate CreditIQ project overview PDF for portfolio / submission use.

Run from project root:
    python scripts/generate_project_overview_pdf.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = PROJECT_ROOT / "documents" / "CreditIQ_Project_Overview.pdf"


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontSize=22,
            spaceAfter=14,
            textColor=colors.HexColor("#1e3a5f"),
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4a5568"),
            spaceAfter=20,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=14,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#2563eb"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1e40af"),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontSize=10,
            leading=13,
            leftIndent=12,
            spaceAfter=4,
        ),
    }


def _table(data: list[list[str]], col_widths: list[float] | None = None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def build_story(s):
    story = []

    story.append(Paragraph("CreditIQ", s["title"]))
    story.append(
        Paragraph(
            "AI-Powered Credit Risk Underwriting Platform<br/>"
            "Detailed Project Overview",
            s["subtitle"],
        )
    )
    story.append(
        Paragraph(
            f"<i>Home Credit Default Risk · LightGBM · SHAP · FastAPI · Streamlit</i><br/>"
            f"Generated {date.today().isoformat()}",
            s["subtitle"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # 1. Executive summary
    story.append(Paragraph("1. Executive Summary", s["h1"]))
    story.append(
        Paragraph(
            "CreditIQ is an end-to-end credit risk intelligence platform built on the Home Credit "
            "Default Risk dataset. It supports loan default prediction, risk band assignment, "
            "explainable AI, business rule transparency, portfolio analytics, and natural-language "
            "portfolio querying — all within a single enterprise Streamlit workspace backed by a "
            "FastAPI inference API.",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "The platform addresses a core underwriting challenge: traditional credit decisions are "
            "often slow, inconsistent, and difficult to explain to business and compliance stakeholders. "
            "CreditIQ combines machine learning with SHAP explainability, structured decision rules, "
            "and executive dashboards to deliver faster, auditable credit decision support.",
            s["body"],
        )
    )

    # 2. Problem statement
    story.append(Paragraph("2. Problem Statement", s["h1"]))
    story.append(
        Paragraph(
            "Home Credit serves a large consumer finance portfolio with an observed default rate of "
            "approximately 8.1%. Manual underwriting does not scale to hundreds of thousands of "
            "applications, and black-box model scores alone are insufficient for regulated or "
            "audit-sensitive credit workflows.",
            s["body"],
        )
    )
    bullets = [
        "Predict probability of loan default at application time",
        "Segment applicants into Low / Medium / High risk bands",
        "Explain which features drive each prediction (SHAP)",
        "Apply transparent business rules aligned with portfolio policy",
        "Monitor portfolio KPIs, approval mix, and risk concentration",
        "Answer portfolio questions via a deterministic AI Data Analyst (no external LLM)",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(b, s["bullet"])) for b in bullets],
            bulletType="bullet",
            start="•",
        )
    )

    # 3. Dataset
    story.append(Paragraph("3. Dataset Overview", s["h1"]))
    story.append(
        _table(
            [
                ["Attribute", "Value"],
                ["Source", "Home Credit Default Risk (application_train)"],
                ["Records", "307,511 labeled applications"],
                ["Target", "TARGET (1 = default, 0 = non-default)"],
                ["Class balance", "91.9% non-default · 8.1% default"],
                ["Raw features", "122 columns in training CSV"],
                ["Model features", "21 engineered inputs"],
            ],
            col_widths=[5.5 * cm, 11 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("3.1 Feature Engineering", s["h2"]))
    story.append(
        Paragraph(
            "Twenty-one model features include external bureau scores (EXT_SOURCE_1/2/3), financial "
            "amounts (income, credit, annuity, goods price), employment and demographic indicators, "
            "and three derived ratios: INCOME_CREDIT_RATIO, ANNUITY_INCOME_RATIO, and CREDIT_GOODS_RATIO. "
            "Median imputation (SimpleImputer, fit on training split) is applied consistently at inference.",
            s["body"],
        )
    )

    # 4. ML pipeline
    story.append(Paragraph("4. Machine Learning Pipeline", s["h1"]))
    story.append(
        _table(
            [
                ["Stage", "Description"],
                ["Dataset", "Home Credit application_train (307,511 rows)"],
                ["Feature engineering", "21 features + 3 financial ratios"],
                ["Imputation", "Median SimpleImputer (training-fit)"],
                ["Model", "LightGBM binary classifier"],
                ["Evaluation", "ROC-AUC, precision, recall, F1 (imbalanced data)"],
                ["Threshold", "0.67 tuned for business objectives"],
                ["Risk bands", "LOW (&lt;0.40) · MEDIUM (0.40–0.67) · HIGH (≥0.67)"],
                ["Decision", "Model + rules → Approve / Review / Decline"],
            ],
            col_widths=[4 * cm, 12.5 * cm],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("4.1 Holdout Model Performance", s["h2"]))
    story.append(
        _table(
            [
                ["Metric", "Value"],
                ["ROC-AUC", "0.7516"],
                ["Accuracy", "0.8620"],
                ["Precision", "0.2566"],
                ["Recall", "0.3742"],
                ["F1 Score", "0.3045"],
                ["Decision threshold", "0.67"],
            ],
            col_widths=[5 * cm, 4 * cm],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "<b>Model selection rationale:</b> LightGBM was chosen for strong tabular performance, "
            "efficient training on large datasets, native handling of heterogeneous financial features, "
            "and compatibility with SHAP-based explainability workflows.",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Class imbalance strategy:</b> Natural 8.1% default rate was preserved (no aggressive "
            "oversampling). Accuracy was not used as the sole metric; threshold tuning balances approval "
            "volume with high-risk detection.",
            s["body"],
        )
    )

    story.append(PageBreak())

    # 5. Architecture
    story.append(Paragraph("5. System Architecture", s["h1"]))
    story.append(
        Paragraph(
            "CreditIQ uses a hybrid architecture: Streamlit provides the UI shell; FastAPI serves "
            "applicant-level inference; portfolio batch scoring and EDA run locally within the "
            "Streamlit process.",
            s["body"],
        )
    )
    story.append(Paragraph("5.1 Component Overview", s["h2"]))
    story.append(
        _table(
            [
                ["Layer", "Technology", "Role"],
                ["Frontend", "Streamlit", "Login, dashboard, EDA, risk, XAI, rules, AI analyst"],
                ["API", "FastAPI", "/predict, /decision, /rules, /health, /dashboard/summary"],
                ["ML", "LightGBM", "Default probability and risk bands"],
                ["Explainability", "SHAP", "Global + local feature contributions"],
                ["Rules", "Python rule engine", "7 structured underwriting policies (R001–R007)"],
                ["Portfolio", "Batch scoring", "Cached snapshot JSON for dashboard KPIs"],
                ["Analyst DB", "SQLite (in-memory)", "NL→SQL portfolio queries"],
            ],
            col_widths=[3 * cm, 3.5 * cm, 9.5 * cm],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("5.2 Frontend ↔ Backend Communication", s["h2"]))
    story.append(
        Paragraph(
            "<b>HTTP (FastAPI):</b> Risk Prediction page POSTs to <i>/decision</i>; Decision Rules "
            "page GETs <i>/rules</i>. Base URL from CREDIT_RISK_API_URL (default http://127.0.0.1:8000; "
            "Docker: http://backend:8000).",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Local Python imports:</b> Executive Dashboard calls build_dashboard_summary() and "
            "load_global_importance() directly. Data Explorer and AI Data Analyst read CSV via "
            "portfolio_loader. Explainability uses preprocess_applicant() locally for SHAP contributions.",
            s["body"],
        )
    )

    # 6. Application modules
    story.append(Paragraph("6. Application Modules", s["h1"]))
    modules = [
        ("Executive Dashboard", "Portfolio KPIs, four analytics charts, SHAP drivers, recent assessments."),
        ("Data Explorer", "Interactive EDA: demographics, financials, risk, data quality (Plotly)."),
        ("Risk Prediction", "Applicant form → API scoring → probability, band, recommendation."),
        ("Explainability", "Local SHAP drivers, contribution chart, narrative summary."),
        ("Decision Rules", "Policy catalog R001–R007, live rule matching, business-readable conditions."),
        ("AI Data Analyst", "Deterministic NL→SQL; no OpenAI/Gemini/RAG; in-memory SQLite."),
        ("Login", "Session gate with demo credentials; enterprise dark UI navigation."),
    ]
    for name, desc in modules:
        story.append(Paragraph(f"<b>{name}</b> — {desc}", s["body"]))

    # 7. API
    story.append(Paragraph("7. FastAPI Endpoints", s["h1"]))
    story.append(
        _table(
            [
                ["Method", "Path", "Description"],
                ["GET", "/health", "Health check"],
                ["POST", "/predict", "Default probability, risk band, model decision"],
                ["POST", "/decision", "Prediction + matched rules + recommendation"],
                ["GET", "/rules", "Active rules catalog"],
                ["GET", "/dashboard/summary", "Executive dashboard payload"],
            ],
            col_widths=[2 * cm, 4.5 * cm, 9.5 * cm],
        )
    )

    # 8. Portfolio metrics
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("8. Portfolio Metrics (Batch-Scored Book)", s["h1"]))
    story.append(
        _table(
            [
                ["Metric", "Value"],
                ["Applications scored", "307,511"],
                ["Observed default rate", "8.1%"],
                ["Approval rate (policy)", "56.2%"],
                ["High-risk exposure (HIGH band)", "12.0%"],
            ],
            col_widths=[6 * cm, 4 * cm],
        )
    )

    # 9. Decision flow
    story.append(Paragraph("9. Decision & Rules Logic", s["h1"]))
    story.append(
        Paragraph(
            "Underwriting rules complement the LightGBM score. Rules are derived from predicted default "
            "probability, risk band thresholds, portfolio policies, and financial risk indicators. "
            "Typical band-to-action mapping: Low → Approve, Medium → Review, High → Decline.",
            s["body"],
        )
    )
    story.append(
        _table(
            [
                ["Risk Score", "Band", "Decision", "Reason (illustrative)"],
                ["22", "Low", "Approve", "Low predicted default probability"],
                ["54", "Medium", "Review", "Moderate risk — manual assessment"],
                ["81", "High", "Decline", "High default probability"],
            ],
            col_widths=[2.5 * cm, 2.5 * cm, 2.5 * cm, 8.5 * cm],
        )
    )

    # 10. AI Data Analyst
    story.append(Paragraph("10. AI Data Analyst (No External LLM)", s["h1"]))
    story.append(
        Paragraph(
            "Natural language questions map to supported business intents via deterministic pattern "
            "matching. Each intent resolves to a predefined SQL template executed against in-memory "
            "SQLite (application_train + portfolio KPIs). Benefits: no hallucinations, consistent "
            "results, fast execution, fully explainable behaviour, zero token/API cost.",
            s["body"],
        )
    )

    story.append(PageBreak())

    # 11. Deployment
    story.append(Paragraph("11. Deployment", s["h1"]))
    story.append(Paragraph("11.1 Local Execution", s["h2"]))
    story.append(
        Paragraph(
            "Terminal 1: uvicorn src.api.main:app --reload --port 8000<br/>"
            "Terminal 2: streamlit run ui/streamlit_app.py<br/>"
            "Set PYTHONPATH=. and place application_train.csv in data/.",
            s["body"],
        )
    )
    story.append(Paragraph("11.2 Docker (Validated Locally)", s["h2"]))
    story.append(
        Paragraph(
            "docker compose up --build starts creditiq-backend (port 8000) and creditiq-frontend "
            "(port 8501). Frontend waits for backend health check. Volumes mount ./data and ./models.",
            s["body"],
        )
    )
    story.append(
        _table(
            [
                ["Service", "URL"],
                ["Streamlit UI", "http://localhost:8501"],
                ["FastAPI", "http://localhost:8000"],
                ["API Docs", "http://localhost:8000/docs"],
            ],
            col_widths=[5 * cm, 11.5 * cm],
        )
    )

    # 12. Tech stack & artifacts
    story.append(Paragraph("12. Technology Stack & Artifacts", s["h1"]))
    story.append(
        _table(
            [
                ["Category", "Components"],
                ["Languages", "Python 3.11"],
                ["ML", "LightGBM, scikit-learn, SHAP, numpy, pandas"],
                ["API", "FastAPI, uvicorn, pydantic"],
                ["UI", "Streamlit, Altair, Plotly"],
                ["Data", "SQLite (in-memory), CSV portfolio"],
                ["Artifacts", "model.pkl, imputer.pkl, metrics.json, shap_values.npy, portfolio_scoring_snapshot.json"],
            ],
            col_widths=[4 * cm, 12.5 * cm],
        )
    )

    # 13. Project status
    story.append(Paragraph("13. Project Status", s["h1"]))
    story.append(
        Paragraph(
            "Feature complete for demonstration and portfolio use. All core modules — Dashboard, "
            "Data Explorer, Risk Prediction, Explainability, Decision Rules, AI Data Analyst, Login, "
            "and Docker deployment — are implemented and validated.",
            s["body"],
        )
    )

    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "<i>Repository: github.com/Nailasalim/credit-risk-intelligence-platform</i>",
            s["subtitle"],
        )
    )

    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="CreditIQ Project Overview",
        author="Nailasalim",
    )
    styles = _styles()
    doc.build(build_story(styles))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
