# Cloud Resource Forecasting — Vercel Edition

This is a Vercel-compatible adaptation of the Streamlit project. Vercel does not run the existing Streamlit server as a normal persistent process, so this version uses a static HTML/JavaScript frontend plus a FastAPI Python function.

## Features
- Interactive user-input scenario calculator (CPU, memory, users, requests/sec, tier thresholds and illustrative hourly rates).
- CSV upload endpoint that runs the existing data-cleaning, chronological ML training/evaluation, and latest-server prediction pipeline.
- API health check at `/api/health`.

## Deploy
1. Put the contents of this folder at the root of a GitHub repository (so `index.html`, `ml_pipeline.py`, `requirements.txt`, `vercel.json`, and `api/index.py` are at the repository root / API folder as shown).
2. Sign in at https://vercel.com/ and choose **Add New → Project**.
3. Import the GitHub repository.
4. Leave Framework Preset as **Other** unless Vercel auto-detects another framework. Keep the root directory at the repository root.
5. Deploy. Vercel installs packages from `requirements.txt` for the Python function.
6. Open the deployment URL and test the scenario calculator, then `/api/health`.

## Dataset format
CSV requires columns `timestamp`, `server_id`, and `cpu_utilization`. Five-minute forecasts need repeated measurements aligned at five-minute intervals per server. Keep the uploaded CSV below 4 MB because serverless request payload limits apply.

## Important limits
- This is not the original Streamlit UI; it is a small web frontend backed by FastAPI, adapted for Vercel's serverless model.
- Python functions have execution, memory, and payload constraints. Scikit-learn/pandas cold starts and model training may exceed limits on some Vercel plans/datasets. If this happens, host the ML API on EC2 or another persistent Python host and keep only the frontend on Vercel.
- The sample API currently permits cross-origin requests broadly, though this single-origin frontend calls it same-origin. Tighten CORS if you later expose the API to other sites.
- Scenario pricing is illustrative, not live AWS pricing. Do not upload sensitive datasets to a public demo.
