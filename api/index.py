from __future__ import annotations
import io
import math
import os
import sys
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make the project root importable in Vercel's Python function environment.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml_pipeline import clean_dataset, train_and_evaluate, predict_latest, cost_summary, data_quality_summary

app = FastAPI(title="Cloud Resource Forecasting API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ScenarioInput(BaseModel):
    cpu: float = Field(ge=0, le=100)
    memory: float = Field(ge=0, le=100)
    active_users: int = Field(ge=0, le=10_000_000)
    requests_per_second: float = Field(ge=0, le=10_000_000)
    small_max: float = Field(ge=10, le=60)
    medium_max: float = Field(ge=61, le=90)
    small_rate: float = Field(ge=0, le=10000)
    medium_rate: float = Field(ge=0, le=10000)
    large_rate: float = Field(ge=0, le=10000)

@app.get("/")
def root():
    return {"message": "Cloud Resource Forecasting API", "health": "/api/health"}

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "cloud-resource-forecasting"}

@app.post("/api/scenario")
def evaluate_scenario(payload: ScenarioInput):
    if payload.small_max >= payload.medium_max:
        raise HTTPException(status_code=422, detail="Small tier CPU limit must be lower than Medium tier CPU limit.")
    if payload.cpu < payload.small_max:
        tier, rate = "Small", payload.small_rate
    elif payload.cpu < payload.medium_max:
        tier, rate = "Medium", payload.medium_rate
    else:
        tier, rate = "Large", payload.large_rate
    if payload.cpu >= 85 or payload.memory >= 90:
        risk, guidance = "High", "Consider scaling up or investigating the workload; verify with sustained measurements."
    elif payload.cpu >= payload.medium_max or payload.memory >= 80:
        risk, guidance = "Elevated", "Monitor subsequent measurements and consider a larger tier if the load persists."
    else:
        risk, guidance = "Normal", "Entered values are below the scenario's high-utilization guardrails."
    return {
        "tier": tier,
        "risk": risk,
        "hourly_cost_usd": round(rate, 4),
        "cost_30_days_usd": round(rate * 24 * 30, 2),
        "active_users": payload.active_users,
        "requests_per_second": payload.requests_per_second,
        "guidance": guidance,
        "note": "Rule-based scenario only; not a trained-model prediction or live AWS pricing. User count and request rate provide context and do not change the tier formula.",
    }

@app.post("/api/forecast")
async def forecast(file: UploadFile = File(...), server_id: str | None = None):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    raw_bytes = await file.read()
    # Vercel functions have request/body limits. Keep uploads modest for serverless use.
    if len(raw_bytes) > 4_000_000:
        raise HTTPException(status_code=413, detail="CSV is too large for this demo endpoint. Please upload a file smaller than 4 MB.")
    try:
        raw = pd.read_csv(io.BytesIO(raw_bytes))
        cleaned, notes = clean_dataset(raw)
        result = train_and_evaluate(cleaned)
        servers = sorted(cleaned["server_id"].dropna().astype(str).unique().tolist())
        selected_server = server_id if server_id in servers else servers[0]
        latest = predict_latest(cleaned, result, selected_server)
        metrics = result.get("test_metrics", {})
        comparison = result.get("comparison", pd.DataFrame()).replace({float("inf"): None, float("-inf"): None}).to_dict(orient="records")
        quality = data_quality_summary(raw, cleaned)
        costs = cost_summary(result.get("test_predictions")["predicted"].to_numpy())
        return {
            "filename": file.filename,
            "rows_uploaded": int(len(raw)),
            "rows_cleaned": int(len(cleaned)),
            "servers": servers,
            "selected_server": selected_server,
            "best_model": result.get("best_model_name"),
            "metrics": {k: (float(v) if v is not None and math.isfinite(float(v)) else None) for k, v in metrics.items()},
            "comparison": comparison,
            "latest_prediction": latest,
            "quality": quality,
            "cost_summary": costs,
            "cleaning_notes": notes,
            "warning": "Metrics and forecast quality depend on the uploaded dataset. Cost rates are illustrative and are not AWS quotes.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
