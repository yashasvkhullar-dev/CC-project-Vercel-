from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_COLUMNS = ["timestamp", "server_id", "cpu_utilization"]
OPTIONAL_NUMERIC = [
    "memory_utilization", "storage_utilization", "network_traffic",
    "active_users", "request_rate", "temperature"
]
NUMERIC_CANDIDATES = ["cpu_utilization"] + OPTIONAL_NUMERIC
CATEGORICAL_CANDIDATES = ["server_id", "workload_level"]
LAG_MINUTES = [5, 10, 15]
CLOUD_PRICING = {"Small": 0.020, "Medium": 0.040, "Large": 0.080}
HOURS_PER_MONTH = 730


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and normalize uploaded telemetry without assuming every optional field exists."""
    if raw is None or raw.empty:
        raise ValueError("The uploaded CSV is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = raw.copy()
    notes = []
    original_rows = len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df["server_id"] = df["server_id"].astype("string").str.strip()
    df.loc[df["server_id"].isin(["", "nan", "None", "<NA>"]), "server_id"] = pd.NA
    df = df.dropna(subset=["timestamp", "server_id"]).copy()
    notes.append(f"Removed {original_rows - len(df)} rows with invalid timestamps or server IDs.")

    # Normalize identifiers like S1, server-001, server_2 into S001, S002, etc. if numeric suffix exists.
    suffix = df["server_id"].astype(str).str.extract(r"(\d+)$", expand=False)
    normalized = suffix.notna()
    df.loc[normalized, "server_id"] = "S" + suffix[normalized].str.zfill(3)

    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype("string").str.replace("%", "", regex=False).str.strip(),
                errors="coerce"
            )

    if "workload_level" in df.columns:
        df["workload_level"] = (
            df["workload_level"].astype("string").str.strip().str.lower()
            .replace({"med": "medium", "hig": "high", "cri": "critical", "": pd.NA})
            .fillna("unknown")
        )
    else:
        df["workload_level"] = "unknown"

    before_dupes = len(df)
    df = df.drop_duplicates()
    notes.append(f"Removed {before_dupes - len(df)} exact duplicate rows.")

    if "active_users" in df:
        df.loc[df["active_users"] < 0, "active_users"] = np.nan
    if "temperature" in df:
        df.loc[df["temperature"] <= -100, "temperature"] = np.nan
    for col in ["cpu_utilization", "memory_utilization", "storage_utilization"]:
        if col in df:
            df.loc[~df[col].between(0, 100), col] = np.nan

    # Aggregate duplicate measurements for the same server/time.
    numeric_existing = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    aggregations = {c: "mean" for c in numeric_existing}
    aggregations["workload_level"] = "first"
    df = df.groupby(["server_id", "timestamp"], as_index=False).agg(aggregations)
    df = df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)

    if df["cpu_utilization"].notna().sum() < 8:
        raise ValueError("At least 8 valid CPU measurements are required to build and evaluate a model.")
    if df["server_id"].nunique() < 1:
        raise ValueError("No valid server identifiers remain after cleaning.")

    notes.append("Normalized server IDs, parsed timestamps, converted numeric fields, and aggregated duplicate server/time measurements.")
    return df, notes


def _available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    # Workload is always created by clean_dataset; server_id is required.
    categorical = ["server_id", "workload_level"]
    return numeric, categorical


def build_supervised_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Create target CPU(t+5m) and historical lags by exact server/timestamp joins.
    No target/future-derived column is included in model features.
    """
    if not {"timestamp", "server_id", "cpu_utilization"}.issubset(df.columns):
        raise ValueError("Dataset must be cleaned before feature engineering.")

    base = df.copy().sort_values(["server_id", "timestamp"]).reset_index(drop=True)
    keys = ["server_id", "timestamp"]

    future = base[keys + ["cpu_utilization"]].copy()
    future["timestamp"] = future["timestamp"] - pd.Timedelta(minutes=5)
    future = future.rename(columns={"cpu_utilization": "target_cpu_5min"})
    base = base.merge(future, on=keys, how="left", validate="one_to_one")

    for minutes in LAG_MINUTES:
        history = base[keys + ["cpu_utilization"]].copy()
        history["timestamp"] = history["timestamp"] + pd.Timedelta(minutes=minutes)
        history = history.rename(columns={"cpu_utilization": f"cpu_lag_{minutes}min"})
        base = base.merge(history, on=keys, how="left", validate="one_to_one")

    base["cpu_change_5min"] = base["cpu_utilization"] - base["cpu_lag_5min"]
    base["cpu_change_10min"] = base["cpu_utilization"] - base["cpu_lag_10min"]
    lag_cols = [f"cpu_lag_{m}min" for m in LAG_MINUTES]
    base["cpu_recent_mean"] = base[["cpu_utilization"] + lag_cols].mean(axis=1)
    base["hour"] = base["timestamp"].dt.hour
    base["day_of_week"] = base["timestamp"].dt.dayofweek
    base["hour_sin"] = np.sin(2 * np.pi * base["hour"] / 24)
    base["hour_cos"] = np.cos(2 * np.pi * base["hour"] / 24)

    numeric, categorical = _available_features(base)
    engineered_numeric = [
        "cpu_lag_5min", "cpu_lag_10min", "cpu_lag_15min",
        "cpu_change_5min", "cpu_change_10min", "cpu_recent_mean",
        "hour_sin", "hour_cos", "day_of_week"
    ]
    numeric_features = list(dict.fromkeys(numeric + engineered_numeric))
    categorical_features = categorical
    # Exclude timestamp and target by construction.
    feature_cols = numeric_features + categorical_features
    base = base.dropna(subset=["target_cpu_5min"]).copy()
    base = base.replace([np.inf, -np.inf], np.nan)
    if len(base) < 12:
        raise ValueError(
            "Not enough five-minute-aligned records to train and evaluate. "
            "The dataset needs repeated measurements at five-minute intervals for the same servers."
        )
    return base, numeric_features, categorical_features


def _make_pipeline(numeric_features: list[str], categorical_features: list[str], regressor) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Chronological split; validation chooses model; test is used only for final evaluation."""
    base, numeric_features, categorical_features = build_supervised_dataset(df)
    timestamps = np.sort(base["timestamp"].unique())
    if len(timestamps) < 5:
        raise ValueError("At least five distinct timestamps with future targets are required for chronological evaluation.")

    train_end = timestamps[max(0, int(len(timestamps) * 0.70) - 1)]
    val_end = timestamps[max(0, int(len(timestamps) * 0.85) - 1)]
    train = base[base["timestamp"] <= train_end]
    val = base[(base["timestamp"] > train_end) & (base["timestamp"] <= val_end)]
    test = base[base["timestamp"] > val_end]
    if min(len(train), len(val), len(test)) == 0:
        # For small datasets use a 60/20/20 split, still chronological.
        train_idx = int(len(timestamps) * 0.60)
        val_idx = int(len(timestamps) * 0.80)
        train_end = timestamps[max(0, train_idx - 1)]
        val_end = timestamps[max(train_idx, val_idx - 1)]
        train = base[base["timestamp"] <= train_end]
        val = base[(base["timestamp"] > train_end) & (base["timestamp"] <= val_end)]
        test = base[base["timestamp"] > val_end]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("The dataset is too small for a chronological train/validation/test split. Add more time points.")

    features = numeric_features + categorical_features
    target = "target_cpu_5min"
    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]

    candidates = [("Linear Regression", LinearRegression())]
    candidates.extend((f"Ridge (alpha={alpha:g})", Ridge(alpha=alpha)) for alpha in [0.01, 0.1, 1, 10, 100])
    rows, best_name, best_model, best_mae = [], None, None, float("inf")
    for name, regressor in candidates:
        candidate = _make_pipeline(numeric_features, categorical_features, regressor)
        candidate.fit(X_train, y_train)
        pred = candidate.predict(X_val)
        metric = _metrics(y_val, pred)
        rows.append({"Model": name, **metric})
        if metric["MAE"] < best_mae:
            best_name, best_model, best_mae = name, regressor, metric["MAE"]

    # Final refit on train + validation, preserving the test set for final evaluation.
    train_val = pd.concat([train, val]).sort_values(["timestamp", "server_id"])
    final_model = _make_pipeline(numeric_features, categorical_features, best_model)
    final_model.fit(train_val[features], train_val[target])
    test_predictions = np.clip(final_model.predict(X_test), 0, 100)
    test_metrics = _metrics(y_test, test_predictions)
    pred_df = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "server_id": test["server_id"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": test_predictions,
    }).sort_values(["timestamp", "server_id"]).reset_index(drop=True)

    return {
        "base": base,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": features,
        "best_model_name": best_name,
        "best_regressor": best_model,
        "model": final_model,
        "comparison": pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True),
        "test_metrics": test_metrics,
        "test_predictions": pred_df,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
    }


def predict_latest(df: pd.DataFrame, result: dict, server_id: str) -> dict:
    base = build_supervised_dataset_for_forecast(df)
    server_rows = base[base["server_id"].astype(str) == str(server_id)].sort_values("timestamp")
    if server_rows.empty:
        raise ValueError("No usable observations for this server.")
    latest = server_rows.tail(1)
    features = result["feature_columns"]
    pred = float(np.clip(result["model"].predict(latest[features])[0], 0, 100))
    return {
        "current_cpu": float(latest["cpu_utilization"].iloc[0]),
        "prediction": pred,
        "forecast_time": latest["timestamp"].iloc[0] + pd.Timedelta(minutes=5),
    }


def build_supervised_dataset_for_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Build the same features without requiring the future target to exist."""
    base = df.copy().sort_values(["server_id", "timestamp"]).reset_index(drop=True)
    keys = ["server_id", "timestamp"]
    for minutes in LAG_MINUTES:
        history = base[keys + ["cpu_utilization"]].copy()
        history["timestamp"] = history["timestamp"] + pd.Timedelta(minutes=minutes)
        history = history.rename(columns={"cpu_utilization": f"cpu_lag_{minutes}min"})
        base = base.merge(history, on=keys, how="left", validate="one_to_one")
    base["cpu_change_5min"] = base["cpu_utilization"] - base["cpu_lag_5min"]
    base["cpu_change_10min"] = base["cpu_utilization"] - base["cpu_lag_10min"]
    base["cpu_recent_mean"] = base[["cpu_utilization"] + [f"cpu_lag_{m}min" for m in LAG_MINUTES]].mean(axis=1)
    base["hour"] = base["timestamp"].dt.hour
    base["day_of_week"] = base["timestamp"].dt.dayofweek
    base["hour_sin"] = np.sin(2 * np.pi * base["hour"] / 24)
    base["hour_cos"] = np.cos(2 * np.pi * base["hour"] / 24)
    return base.replace([np.inf, -np.inf], np.nan)


def cost_summary(predictions, rates=None, small_max=35, medium_max=70) -> dict:
    rates = rates or CLOUD_PRICING
    pred = np.clip(np.asarray(predictions, dtype=float).reshape(-1), 0, 100)
    if pred.size == 0:
        raise ValueError("No test predictions available for cost estimation.")
    if not (0 < small_max < medium_max < 100):
        raise ValueError("CPU thresholds must satisfy 0 < small threshold < medium threshold < 100.")

    tiers = np.where(pred < small_max, "Small", np.where(pred < medium_max, "Medium", "Large"))
    interval_hours = 5 / 60
    fixed_cost = float(len(pred) * interval_hours * rates["Large"])
    scaled_cost = float(sum(rates[tier] for tier in tiers) * interval_hours)
    savings = fixed_cost - scaled_cost
    savings_pct = (savings / fixed_cost * 100) if fixed_cost else 0.0
    total_hours = len(pred) * interval_hours
    monthly_fixed = fixed_cost / total_hours * HOURS_PER_MONTH if total_hours else 0.0
    monthly_scaled = scaled_cost / total_hours * HOURS_PER_MONTH if total_hours else 0.0
    return {
        "intervals": int(len(pred)), "total_hours": total_hours,
        "fixed_cost": fixed_cost, "scaled_cost": scaled_cost,
        "savings": savings, "savings_pct": savings_pct,
        "monthly_savings": monthly_fixed - monthly_scaled,
        "hours_per_month": HOURS_PER_MONTH,
        "tier_counts": {tier: int(np.sum(tiers == tier)) for tier in ["Small", "Medium", "Large"]},
    }


def data_quality_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    return {
        "Original row count": f"{len(raw):,}",
        "Cleaned row count": f"{len(cleaned):,}",
        "Rows removed during cleaning": f"{len(raw) - len(cleaned):,}",
        "Distinct servers": f"{cleaned['server_id'].nunique():,}",
        "Missing CPU values": f"{cleaned['cpu_utilization'].isna().sum():,}",
        "Duplicate server/timestamp keys": f"{cleaned.duplicated(['server_id', 'timestamp']).sum():,}",
        "Missing values (all columns)": f"{int(cleaned.isna().sum().sum()):,}",
    }
