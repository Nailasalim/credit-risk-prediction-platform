# Hosting CreditIQ (public URL — no local `cmd` for visitors)

CreditIQ needs **Streamlit + FastAPI**, **`models/`**, **`data/application_train.csv`**, and **`GROQ_API_KEY`**.

This guide uses **one Docker service** that runs both apps (`scripts/start_hosted.sh`).

Recommended platforms: **[Railway](https://railway.app)** (easiest) or **[Render](https://render.com)**.

---

## Before you deploy

### 1. Push code to GitHub

```powershell
git add scripts/start_hosted.sh render.yaml railway.toml Dockerfile README.md
git add models/ ui/ src/ requirements.txt .streamlit/
git status   # .env and data/*.csv must NOT appear
git commit -m "Add hosted deploy config for Railway/Render"
git push origin main
```

### 2. Confirm `models/` is on GitHub

The remote must include at least:

- `models/model.pkl`
- `models/imputer.pkl`
- `models/label_encoders.pkl`
- `models/feature_names.json`
- `models/metrics.json`
- `models/shap_values.npy`
- `models/portfolio_scoring_snapshot.json` (optional but recommended)

If `models/` is missing on GitHub, add and push it (files can be large).

### 3. Plan for the CSV

`application_train.csv` is **not** in Git (too large). On the host you must either:

- Upload it to a **volume/disk** at `/app/data/application_train.csv`, or  
- Use a one-time upload / S3 / Drive download (advanced)

---

## Option A — Railway (recommended)

1. Go to [railway.app](https://railway.app) → sign in with GitHub  
2. **New Project** → **Deploy from GitHub repo** → select `credit-risk-prediction-platform`  
3. Railway detects the `Dockerfile`  
4. **Settings → Deploy**  
   - Start Command: `bash scripts/start_hosted.sh`  
   (or leave blank if `railway.toml` is picked up)  
5. **Variables** → add:

| Variable | Value |
|----------|--------|
| `GROQ_API_KEY` | your Groq key |
| `CREDIT_RISK_API_URL` | `http://127.0.0.1:8000` |
| `CREDIT_RISK_PORTFOLIO_CSV` | `/app/data/application_train.csv` |
| `PYTHONPATH` | `/app` |

6. **Volume** (for the CSV):  
   - Add a volume mounted at `/app/data`  
   - Upload `application_train.csv` into that volume (Railway CLI / dashboard file browser / one-shot shell)

   Example with Railway CLI after linking the project:

   ```powershell
   railway volume list
   # Or open a shell on the service and upload via scp/curl from a public URL you control
   ```

   Practical approach for students: host the CSV temporarily on Google Drive / Dropbox as a **direct download link**, then in Railway **Shell**:

   ```bash
   mkdir -p /app/data
   curl -L "YOUR_DIRECT_CSV_URL" -o /app/data/application_train.csv
   ```

7. **Settings → Networking → Generate Domain**  
   You get a URL like `https://creditiq-production.up.railway.app`

8. Open that URL → login:

| User | Password |
|------|----------|
| `analyst` | `AnalystDemo2024` |

---

## Option B — Render

1. [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**  
2. Connect the GitHub repo (uses `render.yaml`)  
3. Set **`GROQ_API_KEY`** when prompted (`sync: false`)  
4. Add a **Disk** mounted at `/app/data` and upload `application_train.csv`  
5. Deploy → open the `.onrender.com` URL  

Free tier **sleeps** after idle time; first load can take ~1 minute.

---

## How the hosted process works

```
Internet → Streamlit (:$PORT)
                │
                └── HTTP → FastAPI (127.0.0.1:8000)  [same container]
```

`scripts/start_hosted.sh` starts both. Visitors only need the HTTPS link.

---

## After deploy checklist

- [ ] Site opens over HTTPS  
- [ ] Login works (`analyst` / `AnalystDemo2024`)  
- [ ] Dashboard loads (may take 30–60s first time)  
- [ ] Risk Prediction scores an applicant  
- [ ] AI Data Analyst works (needs valid `GROQ_API_KEY`)  

---

## Common issues

| Problem | Fix |
|---------|-----|
| Dashboard / EDA error about CSV | Upload `application_train.csv` to `/app/data/` |
| Model load error | Ensure `models/*.pkl` are in the GitHub repo / image |
| AI Analyst fails | Set `GROQ_API_KEY` on the host |
| App sleeps (Render free) | Wait for cold start, or upgrade plan |
| Build OOM | Ensure CSV is **not** copied into the Docker image |

---

## Security note

Demo login is **not** real security. Anyone with the URL can use the demo accounts. Fine for a portfolio demo; rotate `GROQ_API_KEY` if abused.

---

## Local test of the hosted start script (optional)

```powershell
docker build -t creditiq-hosted .
docker run --rm -p 8501:8501 ^
  -e PORT=8501 ^
  -e GROQ_API_KEY=your_key ^
  -v ${PWD}/data:/app/data:ro ^
  -v ${PWD}/models:/app/models:ro ^
  creditiq-hosted bash scripts/start_hosted.sh
```

Then open http://localhost:8501
