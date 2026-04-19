"""
EnergyX — Probabilistic Forecast Evaluation
=============================================
Standalone test script validating prediction interval (PI) methods before
they are integrated into the main benchmarks.

Methods
-------
1. ARIMA     — pmdarima analytic 90% CI via `predict(return_conf_int=True, alpha=0.10)`
2. ETS       — Bootstrap simulation: fit.simulate(500 paths), Q5/Q95
3. LGBM-Q    — Dual quantile models (alpha=0.05 / 0.95) with recursive point forecast
               for lag updates. NOTE: known to underestimate uncertainty at long horizons
               because lag errors don't compound through quantile networks.
4. Residual  — Model-agnostic empirical residuals from 3-fold expanding-window CV.
               Per-step Q5/Q95 of residuals added to point forecast.
               Works for ANY model (Naive, RF, N-BEATS, …).

Metrics
-------
PICP    — Prediction Interval Coverage Probability  (target: 0.90)
MIW     — Mean Interval Width                       (smaller = sharper)
PINAW   — Normalised Average Width = MIW / range(y_train)  (scale-free)
Winkler — (u-l) + (2/α)*[max(l-y,0) + max(y-u,0)] for α=0.10
           Penalises miscoverage; lower = better calibrated AND sharp

Datasets evaluated
------------------
ASHRAE_elec, ASHRAE_chilled  (target: meter_reading, hourly, sp=24)
ETTh1, ETTh2                 (target: OT, hourly, sp=24)
ETTm1, ETTm2                 (target: OT, 15-min, sp=96)

For ETT: first slice only (first 512 train + 96 test points) so the script
runs fast while still being representative.

Usage
-----
    source venv/bin/activate
    python evaluate_probabilistic.py
    python evaluate_probabilistic.py --fast          # ASHRAE only, H=24
    python evaluate_probabilistic.py --horizons 24 96
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_forecasting import (
    _create_lag_features,
    _lag_col_names,
    _recursive_forecast,
    _fit_predict_lgbm_uni,
    forecast_naive,
    forecast_seasonal_naive,
    forecast_arima,
    _fit_predict_ets,
    ARIMA_CONTEXT_CAP,
)

ALPHA = 0.10          # 90% prediction interval
HORIZONS = [24, 96, 168]
TRAIN_RATIO = 0.80
ETS_SIMS = 500        # bootstrap paths for ETS
CV_FOLDS = 3          # folds for empirical residual intervals
MAX_WINDOWS = 30      # rolling eval windows per horizon (keep fast)

ASHRAE_TARGET = "meter_reading"
ASHRAE_SP = 24
ETT_TARGET = "OT"
ETT_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

DATASETS = {
    "ASHRAE_elec":    ("ashrae", ROOT / "data/raw/building_energy_data (copy 1)_elec.csv",    ASHRAE_TARGET, ASHRAE_SP),
    "ASHRAE_chilled": ("ashrae", ROOT / "data/raw/building_energy_data (copy 1)_chilled.csv", ASHRAE_TARGET, ASHRAE_SP),
    "ETTh1":          ("ett",    ROOT / "data/test/ETTh1.csv",  ETT_TARGET, 24),
    "ETTh2":          ("ett",    ROOT / "data/test/ETTh2.csv",  ETT_TARGET, 24),
    "ETTm1":          ("ett",    ROOT / "data/test/ETTm1.csv",  ETT_TARGET, 96),
    "ETTm2":          ("ett",    ROOT / "data/test/ETTm2.csv",  ETT_TARGET, 96),
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_series(kind: str, path: Path, target: str) -> np.ndarray:
    df = pd.read_csv(path)
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "ts"})
    time_col = "date" if "date" in df.columns else "timestamp"
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    df = df.dropna(subset=[target])
    return df[target].values.astype(float)


# ── PI metric functions ────────────────────────────────────────────────────────

def picp(y_true, lower, upper):
    """Coverage probability (target: 1-alpha = 0.90)."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def miw(lower, upper):
    """Mean interval width."""
    return float(np.mean(upper - lower))


def pinaw(lower, upper, y_train):
    """Normalised average width = MIW / range(y_train)."""
    r = float(y_train.max() - y_train.min())
    return miw(lower, upper) / r if r > 0 else float("inf")


def winkler(y_true, lower, upper, alpha=ALPHA):
    """
    Winkler score: interval width + penalty for observations outside.
    Lower is better.
    """
    width = upper - lower
    pen_lo = (2.0 / alpha) * np.maximum(lower - y_true, 0.0)
    pen_hi = (2.0 / alpha) * np.maximum(y_true - upper, 0.0)
    return float(np.mean(width + pen_lo + pen_hi))


def compute_pi_metrics(y_true, lower, upper, y_train, alpha=ALPHA):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    # Enforce lower <= upper (quantile crossing fix)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    return {
        "PICP":   picp(y_true, lo, hi),
        "MIW":    miw(lo, hi),
        "PINAW":  pinaw(lo, hi, y_train),
        "Winkler": winkler(y_true, lo, hi, alpha),
    }


# ── Method 1: ARIMA analytic CI ───────────────────────────────────────────────

def pi_arima(y_ctx, horizon, alpha=ALPHA):
    """
    ARIMA 90% PI via pmdarima analytic intervals.
    Returns (lower, upper, point_forecast).
    Falls back to empirical residuals on failure.
    """
    y = y_ctx[-ARIMA_CONTEXT_CAP:]
    try:
        import pmdarima as pm
        model = pm.auto_arima(
            y, start_p=0, max_p=5, start_q=0, max_q=5,
            start_d=0, max_d=2, seasonal=False, stepwise=True,
            suppress_warnings=True, error_action="ignore", trace=False,
        )
        fc, ci = model.predict(n_periods=horizon, return_conf_int=True, alpha=alpha)
        return ci[:, 0], ci[:, 1], fc
    except Exception:
        fc = np.full(horizon, y[-1])
        return fc, fc, fc


# ── Method 2: ETS bootstrap simulation ────────────────────────────────────────

def pi_ets(y_ctx, horizon, sp, trend="add", seasonal="add",
           n_sims=ETS_SIMS, alpha=ALPHA):
    """
    ETS 90% PI via bootstrap simulation (resample residuals).
    Returns (lower, upper, point_forecast).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = y_ctx[-ARIMA_CONTEXT_CAP:]
    t = trend if trend != "none" else None
    s = seasonal if seasonal != "none" else None
    if (t == "mul" or s == "mul") and np.any(y <= 0):
        t = "add" if t == "mul" else t
        s = "add" if s == "mul" else s
    try:
        sp_eff = sp if s else None
        model = ExponentialSmoothing(y, trend=t, seasonal=s, seasonal_periods=sp_eff)
        fit = model.fit(optimized=True)
        # Point forecast
        fc = np.asarray(fit.forecast(horizon), dtype=float)
        # Bootstrap simulation: (horizon, n_sims)
        sim = fit.simulate(
            nsimulations=horizon,
            repetitions=n_sims,
            anchor="end",
            random_errors="bootstrap",
        )
        lo = np.percentile(sim, alpha / 2 * 100, axis=1)
        hi = np.percentile(sim, (1 - alpha / 2) * 100, axis=1)
        return lo, hi, fc
    except Exception:
        fc = np.full(horizon, float(y_ctx[-1]))
        return fc, fc, fc


# ── Method 3: LightGBM quantile regression ────────────────────────────────────

def _train_lgbm_quantile_pair(y_train, n_lags, alpha=ALPHA,
                               num_leaves=31, learning_rate=0.1, n_estimators=100):
    """
    Train three LGBM models: point (MSE), lower quantile (alpha/2), upper (1-alpha/2).
    Returns (model_pt, model_lo, model_hi, cols).
    """
    X, y_t = _create_lag_features(y_train, n_lags)
    cols = _lag_col_names(n_lags)
    Xdf = pd.DataFrame(X, columns=cols)

    kw = dict(num_leaves=num_leaves, learning_rate=learning_rate,
              n_estimators=n_estimators, random_state=42, verbose=-1)
    m_pt = lgb.LGBMRegressor(**kw)
    m_lo = lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2, **kw)
    m_hi = lgb.LGBMRegressor(objective="quantile", alpha=1.0 - alpha / 2, **kw)

    m_pt.fit(Xdf, y_t)
    m_lo.fit(Xdf, y_t)
    m_hi.fit(Xdf, y_t)
    return m_pt, m_lo, m_hi, cols


def _recursive_pi_lgbm(m_pt, m_lo, m_hi, last_vals, horizon, cols):
    """
    Recursive forecast: point model updates the lag window each step.
    Quantile models predict lo/hi at each step from the same lag window.

    ⚠ Known limitation: lag errors from point forecast do not propagate
    through the quantile models, so uncertainty is underestimated at
    long horizons. Intervals will be too narrow for H >> n_lags.
    """
    lags = last_vals.copy()
    lo, hi, pt = [], [], []
    for _ in range(horizon):
        row = pd.DataFrame([lags], columns=cols)
        p  = float(m_pt.predict(row)[0])
        l  = float(m_lo.predict(row)[0])
        h  = float(m_hi.predict(row)[0])
        pt.append(p); lo.append(l); hi.append(h)
        lags = np.append(lags[1:], p)   # advance with point forecast
    return np.array(lo), np.array(hi), np.array(pt)


def pi_lgbm_quantile(y_train_ctx, horizon, n_lags=24, alpha=ALPHA,
                      num_leaves=31, learning_rate=0.1, n_estimators=100):
    """Train quantile LGBM on y_train_ctx, forecast horizon steps."""
    try:
        m_pt, m_lo, m_hi, cols = _train_lgbm_quantile_pair(
            y_train_ctx, n_lags, alpha, num_leaves, learning_rate, n_estimators
        )
        lo, hi, pt = _recursive_pi_lgbm(
            m_pt, m_lo, m_hi, y_train_ctx[-n_lags:], horizon, cols
        )
        return lo, hi, pt
    except Exception:
        fc = np.full(horizon, float(y_train_ctx[-1]))
        return fc, fc, fc


# ── Method 4: Empirical residual intervals (model-agnostic) ───────────────────

def _expanding_cv_residuals(predict_fn, y, horizon, n_folds=CV_FOLDS):
    """
    Run expanding-window CV and collect per-step residuals.

    predict_fn: callable(y_train, horizon) -> np.ndarray of length horizon
    Returns: (n_folds * windows) x horizon matrix of signed residuals,
             or None if not enough data.
    """
    n = len(y)
    min_train = max(50, horizon * 2)
    if n < min_train + horizon:
        return None
    step = max(1, (n - min_train - horizon) // n_folds)
    all_res = []
    for i in range(n_folds):
        sp = min_train + i * step
        if sp + horizon > n:
            break
        try:
            preds = np.asarray(predict_fn(y[:sp], horizon), dtype=float)
            actual = y[sp: sp + horizon]
            if len(preds) == len(actual):
                all_res.append(actual - preds)   # residual: positive = under-forecast
        except Exception:
            pass
    return np.array(all_res) if all_res else None  # shape (n_folds, horizon)


def pi_empirical_residuals(predict_fn, y_train, horizon,
                            n_folds=CV_FOLDS, alpha=ALPHA):
    """
    1. Run CV to collect per-step residuals.
    2. Point forecast on full y_train.
    3. PI = point ± per-step quantile of residuals.

    If CV fails, falls back to global residual percentiles.
    """
    residuals = _expanding_cv_residuals(predict_fn, y_train, horizon, n_folds)
    fc = np.asarray(predict_fn(y_train, horizon), dtype=float)

    if residuals is not None and residuals.shape[0] >= 2:
        # Per-step quantiles — shape (horizon,)
        q_lo = np.percentile(residuals, alpha / 2 * 100, axis=0)
        q_hi = np.percentile(residuals, (1 - alpha / 2) * 100, axis=0)
    else:
        # Fallback: global percentiles (scalar shifted interval)
        global_res = _expanding_cv_residuals(
            predict_fn, y_train, min(horizon, 24), n_folds
        )
        if global_res is not None:
            flat = global_res.ravel()
            q_lo = np.full(horizon, np.percentile(flat, alpha / 2 * 100))
            q_hi = np.full(horizon, np.percentile(flat, (1 - alpha / 2) * 100))
        else:
            # Last resort: ±1 std
            std = float(np.std(y_train[-max(horizon, 24):]))
            q_lo = np.full(horizon, -1.65 * std)
            q_hi = np.full(horizon, +1.65 * std)

    return fc + q_lo, fc + q_hi, fc


# ── Rolling evaluation ─────────────────────────────────────────────────────────

def _get_origins(n_test, horizon, max_windows=MAX_WINDOWS):
    total = max(1, (n_test - horizon) // horizon + 1)
    n = min(total, max_windows)
    if total > max_windows:
        idx = np.linspace(0, total - 1, n, dtype=int)
    else:
        idx = np.arange(total)
    return [int(i) * horizon for i in idx]


def evaluate_pi_methods(y_train, y_test, horizon, sp,
                         dataset_name, methods_override=None):
    """
    Run all four PI methods over rolling-origin windows.
    Returns a list of result dicts.
    """
    origins = _get_origins(len(y_test), horizon)
    y_range = float(y_train.max() - y_train.min())

    # LGBM quantile: train once on full training set (fixed params)
    n_lags = min(24, len(y_train) // 4)
    try:
        lgbm_pt, lgbm_lo, lgbm_hi, lgbm_cols = _train_lgbm_quantile_pair(
            y_train, n_lags
        )
        lgbm_trained = True
    except Exception:
        lgbm_trained = False

    # Empirical residual baselines: train predict_fn once, apply CV internally
    def _naive_fn(y_ctx, h):   return forecast_naive(y_ctx, h)
    def _snv_fn(y_ctx, h):     return forecast_seasonal_naive(y_ctx, h, sp)
    def _arima_fn(y_ctx, h):   return forecast_arima(y_ctx, h)
    def _lgbm_fn(y_ctx, h):    return _fit_predict_lgbm_uni(y_ctx, h, n_lags=n_lags)

    # Storage: {method_name: {metric: [per-window values]}}
    store = {}
    METHOD_NAMES = [
        "ARIMA_analytic", "ETS_bootstrap",
        "LGBM_quantile",
        "Naive_empirical", "SeasonalNaive_empirical",
        "ARIMA_empirical", "LGBM_empirical",
    ]
    for mn in METHOD_NAMES:
        store[mn] = {"lower": [], "upper": [], "actual": []}

    print(f"     Windows: {len(origins)}", end="", flush=True)
    t0_total = time.perf_counter()

    for wi, origin in enumerate(origins):
        end = origin + horizon
        if end > len(y_test):
            break

        actual = y_test[origin:end]
        y_ctx = np.concatenate([y_train, y_test[:origin]]) if origin > 0 else y_train

        # 1. ARIMA analytic
        lo, hi, _ = pi_arima(y_ctx, horizon)
        store["ARIMA_analytic"]["lower"].append(lo)
        store["ARIMA_analytic"]["upper"].append(hi)
        store["ARIMA_analytic"]["actual"].append(actual)

        # 2. ETS bootstrap
        lo, hi, _ = pi_ets(y_ctx, horizon, sp)
        store["ETS_bootstrap"]["lower"].append(lo)
        store["ETS_bootstrap"]["upper"].append(hi)
        store["ETS_bootstrap"]["actual"].append(actual)

        # 3. LGBM quantile (trained once, recursive per window)
        if lgbm_trained:
            last = y_ctx[-n_lags:]
            lo_a, hi_a, _ = _recursive_pi_lgbm(lgbm_pt, lgbm_lo, lgbm_hi,
                                                  last, horizon, lgbm_cols)
        else:
            lo_a = hi_a = np.full(horizon, float(y_ctx[-1]))
        store["LGBM_quantile"]["lower"].append(lo_a)
        store["LGBM_quantile"]["upper"].append(hi_a)
        store["LGBM_quantile"]["actual"].append(actual)

        # 4–7. Empirical residuals (each predict_fn re-run on y_ctx)
        for tag, fn in [
            ("Naive_empirical",        _naive_fn),
            ("SeasonalNaive_empirical", _snv_fn),
            ("ARIMA_empirical",         _arima_fn),
            ("LGBM_empirical",          _lgbm_fn),
        ]:
            lo, hi, _ = pi_empirical_residuals(fn, y_ctx, horizon)
            store[tag]["lower"].append(lo)
            store[tag]["upper"].append(hi)
            store[tag]["actual"].append(actual)

        if (wi + 1) % max(1, len(origins) // 4) == 0 or wi == len(origins) - 1:
            print(f" [{wi+1}]", end="", flush=True)

    elapsed = time.perf_counter() - t0_total
    print(f"  {elapsed:.1f}s")

    # Aggregate
    rows = []
    for mn in METHOD_NAMES:
        d = store[mn]
        if not d["actual"]:
            continue
        y_true_all = np.concatenate(d["actual"])
        lo_all     = np.concatenate(d["lower"])
        hi_all     = np.concatenate(d["upper"])
        m = compute_pi_metrics(y_true_all, lo_all, hi_all, y_train)
        rows.append({
            "dataset": dataset_name,
            "horizon": horizon,
            "method": mn,
            **m,
        })
    return rows


# ── Pretty print ───────────────────────────────────────────────────────────────

def print_pi_table(rows_for_config):
    """Print PICP / MIW / PINAW / Winkler for one dataset×horizon block."""
    print(
        f"     {'Method':<28s}  {'PICP':>6s}  {'MIW':>9s}"
        f"  {'PINAW':>7s}  {'Winkler':>9s}  Calibrated?"
    )
    print(f"     {'─' * 75}")
    for r in sorted(rows_for_config, key=lambda x: -x["PICP"]):
        ok = "✓" if r["PICP"] >= 0.88 else "✗ under"
        ok = "~ wide" if (r["PICP"] >= 0.88 and r["PINAW"] > 0.5) else ok
        print(
            f"     {r['method']:<28s}  {r['PICP']:6.3f}  {r['MIW']:9.3f}"
            f"  {r['PINAW']:7.4f}  {r['Winkler']:9.3f}  {ok}"
        )


# ── Main ────────────────────────────────────────────────────────────────────────

def run_probabilistic_eval(horizons=None, fast=False):
    if horizons is None:
        horizons = HORIZONS

    datasets = DATASETS
    if fast:
        datasets = {k: v for k, v in DATASETS.items() if k.startswith("ASHRAE")}
        horizons = [24]

    all_rows = []

    for ds_name, (kind, path, target, sp) in datasets.items():
        print(f"\n{'=' * 72}")
        print(f"  Dataset: {ds_name}  (sp={sp})")
        print(f"{'=' * 72}")

        if not path.exists():
            print(f"  ⚠ File not found: {path}  — skipping")
            continue

        y_all = load_series(kind, path, target)

        # For ETT: use first 512+96 rows to keep runtime manageable
        if kind == "ett":
            y_all = y_all[:512 + 96]
            print(f"  ETT: using first {len(y_all)} rows (512 train + 96 test)")

        n = len(y_all)
        split = int(n * TRAIN_RATIO)
        y_train = y_all[:split]
        y_test  = y_all[split:]
        print(f"  Train: {len(y_train)}   Test: {len(y_test)}")

        for horizon in horizons:
            if horizon > len(y_test):
                print(f"  Skipping H={horizon}: test set too short ({len(y_test)} rows)")
                continue
            print(f"\n  ── Horizon H={horizon}")
            rows = evaluate_pi_methods(y_train, y_test, horizon, sp, ds_name)
            all_rows.extend(rows)
            print_pi_table([r for r in rows])

    # ── Save ──────────────────────────────────────────────────────────────────
    if all_rows:
        out_dir = ROOT / "outputs" / "benchmark"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_rows)
        out_path = out_dir / "probabilistic_pi_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nResults saved → {out_path}")
        print_global_summary(df)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def print_global_summary(df: pd.DataFrame):
    """
    Global summary: mean PICP / MIW / PINAW / Winkler per method across all
    dataset × horizon combinations. The key diagnostic for the paper.
    """
    print(f"\n{'=' * 72}")
    print("  GLOBAL SUMMARY — Mean across all datasets × horizons")
    print(f"{'=' * 72}")
    agg = (
        df.groupby("method")[["PICP", "MIW", "PINAW", "Winkler"]]
        .mean()
        .sort_values("Winkler")
    )
    print(
        f"  {'Method':<28s}  {'PICP':>6s}  {'MIW':>9s}"
        f"  {'PINAW':>7s}  {'Winkler':>9s}  Calibrated?"
    )
    print(f"  {'─' * 75}")
    for name, row in agg.iterrows():
        ok = "✓" if row["PICP"] >= 0.88 else "✗ under"
        ok = "~ wide" if (row["PICP"] >= 0.88 and row["PINAW"] > 0.5) else ok
        print(
            f"  {name:<28s}  {row['PICP']:6.3f}  {row['MIW']:9.3f}"
            f"  {row['PINAW']:7.4f}  {row['Winkler']:9.3f}  {ok}"
        )

    print(f"\n  Notes on calibration target (90% PI = PICP ≥ 0.90):")
    print(f"  ✓  = coverage ≥ 88% (within ±2pp tolerance) AND reasonably sharp")
    print(f"  ~ wide = coverage OK but PINAW > 0.5 (intervals too wide to be useful)")
    print(f"  ✗ under = systematic undercoverage — intervals too narrow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnergyX Probabilistic PI Evaluation")
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--fast", action="store_true",
                        help="ASHRAE only, H=24 (quick sanity check)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    df = run_probabilistic_eval(horizons=args.horizons, fast=args.fast)
    print(f"\nTotal time: {time.perf_counter() - t_start:.1f}s")
