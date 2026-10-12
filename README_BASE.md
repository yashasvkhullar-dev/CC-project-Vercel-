# Cloud Resource Forecasting & Cost Optimization System

A Streamlit application built around the provided `final_cc_code.py` project. It supports uploaded cloud telemetry CSV files, data exploration, five-minute-ahead CPU forecasting, chronological model evaluation, and an illustrative compute-tier cost comparison.

## Features

- Cloud telemetry overview and server-wise resource charts
- CSV upload, validation, cleaning, preview, summary statistics, missing-value chart, histogram, and correlation heatmap
- Five-minute CPU forecasting using Linear Regression and Ridge candidates
- Chronological train / validation / test split
- Validation MAE for model selection; test metrics reported separately
- Actual-versus-predicted, scatter, and residual charts
- Illustrative Small / Medium / Large compute-tier comparison

## Expected CSV columns

Required:
- `timestamp`
- `server_id`
- `cpu_utilization`

Optional numeric columns:
- `memory_utilization`
- `storage_utilization`
- `network_traffic`
- `active_users`
- `request_rate`
- `temperature`

Optional categorical column:
- `workload_level`

The app does not ship with the original dataset because it was not available alongside the supplied Python script. Upload the original dataset in the web interface.

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `ml_pipeline.py`, `requirements.txt`, `README.md`, `.gitignore`, and `tests/`.
2. Do not commit secrets or private data. Keep the dataset out of the repository unless you have permission to publish it.
3. Open Streamlit Community Cloud and sign in with GitHub.
4. Create a new app and select your repository and branch.
5. Set the main file path to `app.py`.
6. Deploy the app.
7. Open the generated URL and upload the dataset in the sidebar.
8. Verify each dashboard tab using the checklist below.

## Validation checklist

- [ ] App starts with `streamlit run app.py`
- [ ] Upload a valid CSV with required columns
- [ ] Overview metrics match the uploaded data
- [ ] Data Explorer charts render
- [ ] CPU forecast renders for a selected server
- [ ] Model evaluation shows actual metrics from the uploaded dataset
- [ ] Cost comparison renders and displays the illustrative-pricing disclaimer
- [ ] Invalid or empty CSV produces an understandable message
- [ ] Deployed app works from the public URL

## Methodology notes and limitations

- The prediction target is the CPU value exactly five minutes after the observation for the same server. Rows without a matching future measurement are excluded from supervised training.
- Historical lags are created by joining the same server's prior timestamps. The dataset should have measurements aligned to five-minute intervals for useful forecasts.
- Model evaluation is chronological. Validation MAE selects the candidate model; the final model is refit on train plus validation and evaluated on the held-out test period.
- This is a forecast from uploaded historical telemetry, not live cloud monitoring.
- Cost figures use illustrative USD hourly prices and a simplified CPU threshold policy. They are not actual cloud provider prices, billing data, or production autoscaling recommendations.
- Performance and results depend on the quality, size, and time coverage of the uploaded dataset.

## EC2 deployment (Ubuntu 24.04)

1. Launch an Ubuntu EC2 instance and allow inbound SSH (22) from **My IP**. For a short academic demo, you can also allow inbound TCP 8501; restrict its source to your team's IPs where possible. Direct public HTTP on 8501 is not HTTPS and is not a production security configuration.
2. Connect with EC2 Instance Connect or SSH.
3. Clone the repository and change into the directory containing `app.py`, `ml_pipeline.py`, and `requirements.txt`.
4. Run `chmod +x deploy_ec2.sh && ./deploy_ec2.sh`. The script creates a virtual environment and installs a systemd service named `cpu-forecasting` so the app starts after reboot.
5. Open `http://EC2_PUBLIC_IPV4:8501` in your browser.
6. Upload a CSV containing `timestamp`, `server_id`, and `cpu_utilization` to use ML forecasting. The interactive capacity scenario works without a dataset.

Useful service commands:

```bash
sudo systemctl status cpu-forecasting
sudo journalctl -u cpu-forecasting -n 100 --no-pager
sudo systemctl restart cpu-forecasting
```

### EC2 security and cost notes

- Do not commit private datasets, credentials, or SSH keys to GitHub.
- For a classroom demo, restrict port 8501 to your team IP addresses when possible. For a public/production app, put it behind HTTPS with a reverse proxy and authentication instead of exposing Streamlit directly.
- Confirm your account's current EC2 eligibility and credits in AWS Billing. Instance runtime, EBS storage, and public IPv4 can have charges depending on eligibility and account plan. Stop or terminate resources when finished; terminating deletes the instance and may delete its root volume depending on settings.
