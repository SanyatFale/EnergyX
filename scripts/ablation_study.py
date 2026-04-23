#!/usr/bin/env python
"""Ablation study: forecasting models × homes × time-frame × density × horizon.

Run:
    python scripts/ablation_study.py
    python scripts/ablation_study.py --homes home96 --models Naive ARIMA --quick
    python scripts/ablation_study.py --output results/ablation_custom.csv

Axes:
  - homes:      home96 | home128 | home62
  - time_frame: 7d | 30d | 90d | 180d  (training window size)
  - density:    1 (12s) | 5 (60s) | 25 (5min) | 50 (10min)
  - horizon:    1h | 24h | 3d | 7d  (in units of the resampled step)
  - model:      Naive | SeasonalNaive | ARIMA | ETS | N-BEATS

For each combination we:
  1. Load the full electricity series for the home from HistoricStore.
  2. Downsample by the density stride.
  3. Pick a random evaluation split point (last `horizon` steps = ground truth).
  4. Slice the training window ending at that split point.
  5. Train the model exactly as the production pipeline does.
  6. Compute MAE / RMSE / MAPE against the held-out ground truth.

Results are written to CSV and a summary table printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Silence noisy third-party warnings before any heavy imports
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOMES = ["home96", "home128", "home62"]

# Training window lengths in days
TIME_FRAMES = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "180d": 180,
}

# Downsampling stride (every Nth row of 12-second data)
DENSITIES = {
    "1 (12s)":  1,
    "5 (60s)":  5,
    "25 (5min)": 25,
    "50 (10min)": 50,
}

# Forecast horizons expressed in *resampled* time steps
# After downsampling by stride D, one step = 12*D seconds.
# Steps for 1h / 24h / 3d / 7d:
#   stride=1  → 300 / 7200 / 21600 / 50400   (12s steps)
#   stride=5  → 60  / 1440 / 4320  / 10080   (60s steps)
#   stride=25 → 12  / 288  / 864   / 2016    (5min steps)
#   stride=50 → 6   / 144  / 432   / 1008    (10min steps)
def _horizon_steps(stride: int) -> dict:
    sec_per_step = 12 * stride
    return {
        "1h":  max(1, 3600  // sec_per_step),
        "24h": max(1, 86400 // sec_per_step),
        "3d":  max(1, 259200 // sec_per_step),
        "7d":  max(1, 604800 // sec_per_step),
    }

UV_MODELS = ["Naive", "SeasonalNaive", "ARIMA", "ETS", "N-BEATS"]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    home: str
    time_frame: str
    density_label: str
    stride: int
    horizon_label: str
    horizon_steps: int
    model: str
    n_train: int
    mae: float
    rmse: float
    mape: float
    training_time_s: float
    status: str
    error: str = ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _mae(yt, yp):
    n = min(len(yt), len(yp))
    return float(np.mean(np.abs(yt[:n] - yp[:n])))

def _rmse(yt, yp):
    n = min(len(yt), len(yp))
    return float(np.sqrt(np.mean((yt[:n] - yp[:n]) ** 2)))

def _mape(yt, yp):
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]
    mask = yt != 0
    if not np.any(mask):
        return float("inf")
    v = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)
    return v if np.isfinite(v) else float("inf")


# ---------------------------------------------------------------------------
# Load electricity series for a home (full range)
# ---------------------------------------------------------------------------
def _load_elec(home_id: str) -> Optional[np.ndarray]:
    from energyx.data.historic_store import HistoricStore
    store = HistoricStore()
    df = store.read_ticks(home_id)
    if df is None or df.empty:
        return None
    # Filter electricity sensor
    elec = df[df["sensor_type"].isin(["electricity_apparent", "electricity_real"])]
    if elec.empty:
        # Try any 'value' column without sensor_type filter
        elec = df
    if "value" not in elec.columns:
        return None
    # Sort by timestamp, aggregate duplicates, fill gaps
    if "ts" in elec.columns:
        elec = elec.sort_values("ts").set_index("ts")
    elec = elec[["value"]].dropna()
    elec = elec[~elec.index.duplicated(keep="first")]
    return elec["value"].values.astype(float)


# ---------------------------------------------------------------------------
# Single model training call (mirrors production pipeline exactly)
# ---------------------------------------------------------------------------
def _tool_call(tool_fn, **kwargs) -> dict:
    """Call a LangChain @tool directly, bypassing the wrapper overhead."""
    # LangChain tools expose the original function as .func
    fn = getattr(tool_fn, "func", tool_fn)
    result = fn(**kwargs)
    return json.loads(result) if isinstance(result, str) else result


def _cv_direct(tool_fn, y, cv_folds, horizon, params):
    """CV helper that calls tools without JSON serialisation round-trips."""
    n = len(y)
    min_train = max(50, horizon * 2)
    scores: List[float] = []

    if n < min_train + horizon:
        sp = max(1, n - horizon)
        try:
            r = _tool_call(tool_fn, train_data=json.dumps(y[:sp].tolist()),
                           horizon=min(horizon, n - sp), **params)
            preds = np.array(r["predictions"])
            scores.append(_mape(y[sp:sp + len(preds)], preds))
        except Exception:
            scores.append(float("inf"))
        return scores

    step = max(1, (n - min_train - horizon) // cv_folds)
    for i in range(cv_folds):
        sp = min_train + i * step
        if sp + horizon > n:
            break
        try:
            r = _tool_call(tool_fn, train_data=json.dumps(y[:sp].tolist()),
                           horizon=min(horizon, n - sp), **params)
            preds = np.array(r["predictions"])
            scores.append(_mape(y[sp:sp + len(preds)], preds))
        except Exception:
            scores.append(float("inf"))
    return scores or [float("inf")]


def _train_and_predict(
    model_name: str,
    y_train: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, float]:
    """Return (predictions_array, training_time_seconds).
    Uses the same tool functions as the production pipeline.
    """
    from tinyts.agent_tools import _get_model_tools, MODEL_TEMPLATES, _param_combinations

    uni_tools, _ = _get_model_tools()
    tmpl = MODEL_TEMPLATES[model_name]

    t0 = time.time()
    combos = _param_combinations(tmpl["search_space"]) if tmpl["search_space"] else [tmpl["hyperparameters"]]

    best_score, best_params = float("inf"), tmpl["hyperparameters"]
    for params in combos:
        scores = _cv_direct(uni_tools[model_name], y_train, 3, horizon, params)
        finite = [s for s in scores if np.isfinite(s)]
        ms = np.mean(finite) if finite else float("inf")
        if ms < best_score:
            best_score, best_params = ms, params

    r = _tool_call(uni_tools[model_name],
                   train_data=json.dumps(y_train.tolist()),
                   horizon=horizon, **best_params)
    preds = np.array(r["predictions"])
    tt = time.time() - t0
    return preds, tt


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------
def run_experiment(
    home_id: str,
    full_series: np.ndarray,
    time_frame_label: str,
    tf_days: int,
    density_label: str,
    stride: int,
    horizon_label: str,
    horizon_steps: int,
    model_name: str,
    rng: random.Random,
) -> RunResult:

    # 1. Downsample
    series = full_series[::stride]

    # Steps per day after downsampling (12s * stride per step)
    steps_per_day = int(86400 / (12 * stride))

    # 2. Training window in steps
    train_steps = tf_days * steps_per_day

    # Minimum viable series: train + horizon
    min_required = train_steps + horizon_steps
    if len(series) < min_required:
        return RunResult(
            home=home_id, time_frame=time_frame_label, density_label=density_label,
            stride=stride, horizon_label=horizon_label, horizon_steps=horizon_steps,
            model=model_name, n_train=0, mae=float("nan"), rmse=float("nan"),
            mape=float("nan"), training_time_s=0.0, status="skip",
            error=f"Series too short: {len(series)} < {min_required}",
        )

    # 3. Random evaluation split point: anywhere in the valid range
    max_split = len(series) - horizon_steps
    min_split = train_steps
    if min_split > max_split:
        return RunResult(
            home=home_id, time_frame=time_frame_label, density_label=density_label,
            stride=stride, horizon_label=horizon_label, horizon_steps=horizon_steps,
            model=model_name, n_train=0, mae=float("nan"), rmse=float("nan"),
            mape=float("nan"), training_time_s=0.0, status="skip",
            error="No valid split point",
        )

    split = rng.randint(min_split, max_split)
    y_train = series[split - train_steps: split]
    y_true  = series[split: split + horizon_steps]

    # Replace NaN/Inf in training data
    y_train = np.where(np.isfinite(y_train), y_train, np.nanmedian(y_train) if np.any(np.isfinite(y_train)) else 0.0)
    if len(y_train) < 10 or not np.any(np.isfinite(y_true)):
        return RunResult(
            home=home_id, time_frame=time_frame_label, density_label=density_label,
            stride=stride, horizon_label=horizon_label, horizon_steps=horizon_steps,
            model=model_name, n_train=len(y_train), mae=float("nan"), rmse=float("nan"),
            mape=float("nan"), training_time_s=0.0, status="skip",
            error="Degenerate train/test split",
        )

    # 4. Train & predict
    try:
        preds, tt = _train_and_predict(model_name, y_train, horizon_steps)
    except Exception as e:
        return RunResult(
            home=home_id, time_frame=time_frame_label, density_label=density_label,
            stride=stride, horizon_label=horizon_label, horizon_steps=horizon_steps,
            model=model_name, n_train=len(y_train), mae=float("nan"), rmse=float("nan"),
            mape=float("nan"), training_time_s=0.0, status="error", error=str(e)[:120],
        )

    # 5. Evaluate
    mae  = _mae(y_true, preds)
    rmse = _rmse(y_true, preds)
    mape = _mape(y_true, preds)

    return RunResult(
        home=home_id, time_frame=time_frame_label, density_label=density_label,
        stride=stride, horizon_label=horizon_label, horizon_steps=horizon_steps,
        model=model_name, n_train=len(y_train),
        mae=round(mae, 4), rmse=round(rmse, 4), mape=round(mape, 4),
        training_time_s=round(tt, 2), status="ok",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EnergyX ablation study")
    p.add_argument("--homes",       nargs="+", default=HOMES,
                   choices=HOMES, metavar="HOME")
    p.add_argument("--models",      nargs="+", default=UV_MODELS,
                   choices=UV_MODELS, metavar="MODEL")
    p.add_argument("--time-frames", nargs="+", default=list(TIME_FRAMES.keys()),
                   choices=list(TIME_FRAMES.keys()), metavar="TF")
    p.add_argument("--densities",   nargs="+", default=list(DENSITIES.keys()),
                   choices=list(DENSITIES.keys()), metavar="D")
    p.add_argument("--horizons",    nargs="+", default=["1h", "24h", "3d", "7d"],
                   choices=["1h", "24h", "3d", "7d"], metavar="H")
    p.add_argument("--output",      default="results/ablation_study.csv")
    p.add_argument("--seed",        type=int, default=RANDOM_SEED)
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke-test: 7d+30d frames, stride 5+25, 1h+24h horizons, Naive+ARIMA only")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.quick:
        homes       = [args.homes[0]]
        models      = ["Naive", "ARIMA"]
        time_frames = ["7d", "30d"]
        densities   = ["5 (60s)", "25 (5min)"]
        horizons    = ["1h", "24h"]
        print("🔬 Quick mode — subset of axes")
    else:
        homes       = args.homes
        models      = args.models
        time_frames = args.time_frames
        densities   = args.densities
        horizons    = args.horizons

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count total experiments
    total = len(homes) * len(time_frames) * len(densities) * len(horizons) * len(models)
    print(f"\n{'='*70}")
    print(f"  EnergyX Ablation Study")
    print(f"  Homes: {homes}")
    print(f"  Time frames: {time_frames}")
    print(f"  Densities: {densities}")
    print(f"  Horizons: {horizons}")
    print(f"  Models: {models}")
    print(f"  Total experiments: {total}")
    print(f"{'='*70}\n")

    # Pre-load all home series (slow parquet scan, worth caching)
    print("Loading electricity series from HistoricStore...")
    home_series = {}
    for home_id in homes:
        print(f"  {home_id}...", end="", flush=True)
        s = _load_elec(home_id)
        if s is None or len(s) == 0:
            print(f" MISSING — skipping")
        else:
            home_series[home_id] = s
            days = len(s) * 12 / 86400
            print(f" {len(s):,} pts ({days:.0f} days)")

    if not home_series:
        print("No data found. Run scripts/ingest_ideal.py first.")
        sys.exit(1)

    results: list[RunResult] = []
    done = 0
    t_start = time.time()

    for home_id, full_series in home_series.items():
        for tf_label in time_frames:
            tf_days = TIME_FRAMES[tf_label]
            for d_label in densities:
                stride = DENSITIES[d_label]
                h_map = _horizon_steps(stride)
                for h_label in horizons:
                    if h_label not in h_map:
                        continue
                    h_steps = h_map[h_label]
                    for model in models:
                        done += 1
                        pct = done / total * 100
                        elapsed = time.time() - t_start
                        eta = (elapsed / done) * (total - done) if done > 0 else 0
                        print(
                            f"[{done:>{len(str(total))}}/{total}] {pct:5.1f}% "
                            f"ETA {eta/60:.1f}m | "
                            f"{home_id:8s} {tf_label:5s} {d_label:12s} {h_label:4s} {model}",
                            end="  ", flush=True,
                        )
                        r = run_experiment(
                            home_id, full_series,
                            tf_label, tf_days,
                            d_label, stride,
                            h_label, h_steps,
                            model, rng,
                        )
                        results.append(r)
                        if r.status == "ok":
                            print(f"MAE={r.mae:.2f}  RMSE={r.rmse:.2f}  MAPE={r.mape:.1f}%  t={r.training_time_s:.1f}s")
                        else:
                            print(f"[{r.status}] {r.error}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_path, index=False)
    print(f"\nResults saved → {output_path}  ({len(df)} rows)")

    # ---------------------------------------------------------------------------
    # Summary tables
    # ---------------------------------------------------------------------------
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("No successful runs to summarise.")
        return

    print(f"\n{'='*70}")
    print("  SUMMARY: mean MAPE (%) by model × time_frame")
    print(f"{'='*70}")
    pivot = ok.pivot_table(values="mape", index="model", columns="time_frame", aggfunc="mean")
    pivot = pivot.reindex(columns=[c for c in TIME_FRAMES if c in pivot.columns])
    print(pivot.round(2).to_string())

    print(f"\n{'='*70}")
    print("  SUMMARY: mean MAPE (%) by model × density")
    print(f"{'='*70}")
    pivot2 = ok.pivot_table(values="mape", index="model", columns="density_label", aggfunc="mean")
    print(pivot2.round(2).to_string())

    print(f"\n{'='*70}")
    print("  SUMMARY: mean MAPE (%) by model × horizon")
    print(f"{'='*70}")
    pivot3 = ok.pivot_table(values="mape", index="model", columns="horizon_label", aggfunc="mean")
    pivot3 = pivot3.reindex(columns=[c for c in ["1h", "24h", "3d", "7d"] if c in pivot3.columns])
    print(pivot3.round(2).to_string())

    print(f"\n{'='*70}")
    print("  SUMMARY: mean MAPE (%) by model × home")
    print(f"{'='*70}")
    pivot4 = ok.pivot_table(values="mape", index="model", columns="home", aggfunc="mean")
    print(pivot4.round(2).to_string())

    print(f"\n{'='*70}")
    print("  SUMMARY: mean training time (s) by model")
    print(f"{'='*70}")
    tt = ok.groupby("model")["training_time_s"].mean().sort_values()
    print(tt.round(2).to_string())

    total_elapsed = time.time() - t_start
    print(f"\nTotal wall time: {total_elapsed/60:.1f} min  |  {len(ok)}/{len(df)} successful runs")

    # Also save a rich summary Excel if openpyxl available
    try:
        xls_path = output_path.with_suffix(".xlsx")
        with pd.ExcelWriter(xls_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="raw", index=False)
            pivot.round(2).to_excel(writer, sheet_name="mape_model_x_timeframe")
            pivot2.round(2).to_excel(writer, sheet_name="mape_model_x_density")
            pivot3.round(2).to_excel(writer, sheet_name="mape_model_x_horizon")
            pivot4.round(2).to_excel(writer, sheet_name="mape_model_x_home")
        print(f"Excel summary → {xls_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
