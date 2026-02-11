"""Self-contained agent tools for TinyTS.

Replaces the LangGraph node pipeline with tool-calling functions.
Tools are created via ``create_agent_tools()`` factory, bound to a
dataset session.  Intermediate results live in a shared ``session`` dict.
"""

import json
import logging
import time as _time
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model templates (from task_planner.py)
# ---------------------------------------------------------------------------
MODEL_TEMPLATES = {
    "Naive": {
        "family": "statistical",
        "hyperparameters": {},
        "search_space": {},
        "supports_multivariate": False,
    },
    "SeasonalNaive": {
        "family": "statistical",
        "hyperparameters": {"seasonal_period": 12},
        "search_space": {"seasonal_period": [7, 12, 24, 30]},
        "supports_multivariate": False,
    },
    "ARIMA": {
        "family": "statistical",
        "hyperparameters": {"p": 1, "d": 1, "q": 1},
        "search_space": {},  # auto_arima handles order selection
        "supports_multivariate": False,
    },
    "ETS": {
        "family": "statistical",
        "hyperparameters": {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
        "search_space": {"trend": ["add", "mul", "none"], "seasonal": ["add", "mul", "none"]},
        "supports_multivariate": False,
    },
    "RandomForest": {
        "family": "tree",
        "hyperparameters": {"n_estimators": 100, "max_depth": 10, "n_lags": 12},
        "search_space": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15],
            "n_lags": [6, 12, 24],
        },
        "supports_multivariate": True,
    },
    "LightGBM": {
        "family": "tree",
        "hyperparameters": {
            "num_leaves": 31, "learning_rate": 0.1,
            "n_estimators": 100, "n_lags": 12,
        },
        "search_space": {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [50, 100, 200],
            "n_lags": [6, 12, 24],
        },
        "supports_multivariate": True,
    },
    "N-BEATS": {
        "family": "neural",
        "hyperparameters": {"n_lags": 24, "hidden_size": 64, "epochs": 50, "learning_rate": 0.001},
        "search_space": {"n_lags": [12, 24, 48], "hidden_size": [32, 64, 128], "epochs": [30, 50, 100]},
        "supports_multivariate": False,
    },
    "TinyTimeMixer": {
        "family": "neural",
        "hyperparameters": {},
        "search_space": {},
        "supports_multivariate": False,
    },
}

# ---------------------------------------------------------------------------
# Lazy-loaded model tool references
# ---------------------------------------------------------------------------
_UNI_TOOLS: Optional[Dict] = None
_MV_TOOLS: Optional[Dict] = None


def _get_model_tools():
    global _UNI_TOOLS, _MV_TOOLS
    if _UNI_TOOLS is None:
        from tinyts.tools.statistical import (
            train_naive_forecaster, train_seasonal_naive_forecaster,
            train_arima_forecaster, train_exponential_smoothing,
        )
        from tinyts.tools.tree_based import (
            train_random_forest_forecaster, train_lightgbm_forecaster,
            train_random_forest_multivariate, train_lightgbm_multivariate,
        )
        from tinyts.tools.neural import train_nbeats, train_tiny_time_mixer

        _UNI_TOOLS = {
            "Naive": train_naive_forecaster,
            "SeasonalNaive": train_seasonal_naive_forecaster,
            "ARIMA": train_arima_forecaster,
            "ETS": train_exponential_smoothing,
            "RandomForest": train_random_forest_forecaster,
            "LightGBM": train_lightgbm_forecaster,
            "N-BEATS": train_nbeats,
            "TinyTimeMixer": train_tiny_time_mixer,
        }
        _MV_TOOLS = {
            "RandomForest": train_random_forest_multivariate,
            "LightGBM": train_lightgbm_multivariate,
        }
    return _UNI_TOOLS, _MV_TOOLS


# ---------------------------------------------------------------------------
# Helpers (extracted from training.py)
# ---------------------------------------------------------------------------

def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "mape") -> float:
    if len(y_true) == 0 or len(y_pred) == 0:
        return float("inf")
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    if metric == "mape":
        # Standard MAPE: percentage error relative to actual values
        mask = yt != 0  # Avoid division by zero
        if not np.any(mask):
            return float("inf")
        s = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)
        return s if np.isfinite(s) else float("inf")
    if metric == "mae":
        return float(np.mean(np.abs(yt - yp)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
    return float("inf")


def _param_combinations(search_space: Dict[str, List]) -> List[Dict]:
    if not search_space:
        return [{}]
    keys = list(search_space.keys())
    vals = list(search_space.values())
    combos = [dict(zip(keys, c)) for c in product(*vals)]
    if len(combos) > 10:
        import random
        random.seed(42)
        combos = random.sample(combos, 10)
    return combos


def _cv_univariate(tool_fn, y, cv_folds, horizon, params):
    n = len(y)
    min_train = max(50, horizon * 2)
    scores: List[float] = []

    if n < min_train + horizon:
        sp = max(1, n - horizon)
        try:
            r = json.loads(tool_fn.invoke({
                "train_data": json.dumps(y[:sp].tolist()),
                "horizon": min(horizon, n - sp), **params,
            }))
            scores.append(_compute_metric(y[sp:sp + len(r["predictions"])], np.array(r["predictions"])))
        except Exception:
            scores.append(float("inf"))
        return scores

    step = max(1, (n - min_train - horizon) // cv_folds)
    for i in range(cv_folds):
        sp = min_train + i * step
        if sp + horizon > n:
            break
        try:
            r = json.loads(tool_fn.invoke({
                "train_data": json.dumps(y[:sp].tolist()),
                "horizon": min(horizon, n - sp), **params,
            }))
            preds = np.array(r["predictions"])
            scores.append(_compute_metric(y[sp:sp + len(preds)], preds))
        except Exception:
            scores.append(float("inf"))
    return scores or [float("inf")]


def _cv_multivariate(tool_fn, y, X, cv_folds, horizon, params):
    n = len(y)
    min_train = max(50, horizon * 2)
    scores: List[float] = []
    step = max(1, (n - min_train - horizon) // max(cv_folds, 1))

    for i in range(max(cv_folds, 1)):
        sp = min_train + i * step
        if sp + horizon > n:
            break
        try:
            r = json.loads(tool_fn.invoke({
                "train_features": json.dumps(X[:sp].tolist()),
                "train_target": json.dumps(y[:sp].tolist()),
                "test_features": json.dumps(X[sp:sp + horizon].tolist()),
                "horizon": min(horizon, n - sp),
                **{k: v for k, v in params.items() if k != "n_lags"},
            }))
            preds = np.array(r["predictions"])
            scores.append(_compute_metric(y[sp:sp + len(preds)], preds))
        except Exception:
            scores.append(float("inf"))
    return scores or [float("inf")]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _save_forecast_plot(y_train, preds, all_results, selected, out_dir):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(y_train))), y=y_train.tolist(),
        mode="lines", name="Historical", line=dict(color="black", width=1.5),
    ))
    fx = list(range(len(y_train), len(y_train) + len(preds)))
    fig.add_trace(go.Scatter(
        x=fx, y=preds, mode="lines", name="Forecast",
        line=dict(color="red", width=2, dash="dash"),
    ))
    fig.add_vline(x=len(y_train), line_dash="dot", line_color="gray")
    fig.update_layout(title="Final Forecast", xaxis_title="Time Step",
                      yaxis_title="Value", template="plotly_white", height=500)
    fig.write_html(str(Path(out_dir) / "final_forecast.html"))

    if len(selected) > 1:
        fig2 = go.Figure()
        colors = ["blue", "green", "orange", "purple", "cyan", "magenta"]
        for i, name in enumerate(selected):
            if name in all_results:
                p = all_results[name]["predictions"]
                fig2.add_trace(go.Scatter(
                    x=fx[:len(p)], y=p, mode="lines", name=name,
                    line=dict(width=1, color=colors[i % len(colors)]), opacity=0.6,
                ))
        fig2.add_trace(go.Scatter(
            x=fx, y=preds, mode="lines", name="Ensemble",
            line=dict(color="red", width=2.5, dash="dash"),
        ))
        fig2.update_layout(title="Model Comparison", xaxis_title="Time Step",
                           yaxis_title="Value", template="plotly_white", height=500)
        fig2.write_html(str(Path(out_dir) / "model_comparison.html"))


def _save_anomaly_plot(y_data, result, target_col, out_dir):
    import plotly.graph_objects as go
    labels = np.array(result["anomaly_labels"])
    mask = labels == -1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(y_data))), y=y_data.tolist(),
        mode="lines", name="Data", line=dict(color="black", width=1),
    ))
    if mask.any():
        fig.add_trace(go.Scatter(
            x=np.where(mask)[0].tolist(), y=y_data[mask].tolist(),
            mode="markers", name="Anomalies",
            marker=dict(color="red", size=8, symbol="x"),
        ))
    fig.update_layout(
        title=f"Anomaly Detection - {result['n_anomalies']} anomalies",
        xaxis_title="Time Step", yaxis_title=target_col,
        template="plotly_white", height=500,
    )
    fig.write_html(str(Path(out_dir) / "anomaly_detection.html"))


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def create_agent_tools(
    dataset_path: str,
    time_column: str,
    target_column: str,
    output_dir: str,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[list, dict]:
    """Create agent tools bound to a dataset session.

    Returns (tools_list, session_dict).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Pre-load data
    df = pd.read_csv(dataset_path, parse_dates=[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)
    y_raw = df[target_column].values.astype(float)

    # Trim leading zeros / NaN
    first_nz = 0
    for i, v in enumerate(y_raw):
        if not np.isnan(v) and v != 0:
            first_nz = i
            break
    if first_nz > 0:
        df = df.iloc[first_nz:].reset_index(drop=True)

    y = df[target_column].values.astype(float)
    y = y[~np.isnan(y)]

    # Feature matrix
    feat_cols = [c for c in (feature_columns or []) if c in df.columns]
    X = None
    y_mv = y
    if feat_cols:
        X_raw = df[feat_cols].values
        valid = ~np.any(np.isnan(X_raw), axis=1) & ~np.isnan(df[target_column].values)
        X = X_raw[valid]
        y_mv = df[target_column].values[valid]

    uni_tools, mv_tools = _get_model_tools()

    session: Dict[str, Any] = {
        "profile": None,
        "dataset_summary": None,
        "model_results": {},
        "model_explanations": {},
        "anomaly_results": None,
        "anomaly_explanation": None,
        "ensemble_strategy": None,
        "final_predictions": None,
        "report": None,
        "output_dir": output_dir,
    }

    # ===== TOOL 1: Profile =====
    @tool
    def profile_dataset() -> str:
        """Profile the dataset: statistics, seasonality, stationarity, plots.

        Call this FIRST before any analysis. No arguments needed.

        Returns:
            JSON with n_rows, frequency, has_trend, has_seasonality,
            seasonal_period, missing_pct, outlier_pct, mean, std,
            adf_pvalue, kpss_pvalue, numeric_columns
        """
        from tinyts.nodes.data_profiler import DataProfilerNode
        node = DataProfilerNode()
        state = {
            "dataset_path": dataset_path,
            "time_column": time_column,
            "target_column": target_column,
            "covariates": feat_cols or None,
            "output_dir": output_dir,
        }
        state = node.execute(state)
        profile = state["data_profile"]
        session["profile"] = profile
        session["dataset_summary"] = state["dataset_summary"]

        return json.dumps({
            "n_rows": profile.shape[0],
            "n_cols": profile.shape[1],
            "frequency": profile.inferred_frequency,
            "date_range": list(profile.date_range),
            "has_trend": profile.has_trend,
            "has_seasonality": profile.has_seasonality,
            "seasonal_period": profile.seasonal_period,
            "missing_pct": round(profile.missing_pct, 2),
            "outlier_pct": round(profile.outlier_pct, 2),
            "mean": round(profile.mean, 4),
            "std": round(profile.std, 4),
            "adf_pvalue": round(profile.adf_pvalue, 4),
            "kpss_pvalue": round(profile.kpss_pvalue, 4),
            "numeric_columns": profile.numeric_columns,
            "feature_columns_available": feat_cols,
        })

    # ===== TOOL 2: Train Forecast Model =====
    @tool
    def train_forecast_model(model_name: str, horizon: int) -> str:
        """Train a single forecasting model with cross-validation.

        Auto-uses multivariate mode for RandomForest/LightGBM when
        feature columns are configured. Results stored internally.

        Args:
            model_name: Choose from available models.
                UNIVARIATE: Naive|SeasonalNaive|ARIMA|ETS|N-BEATS|TinyTimeMixer
                MULTIVARIATE (requires features): RandomForest|LightGBM
            horizon: Number of time steps to forecast

        Returns:
            JSON with model_name, mean_mape, std_mape, best_params,
            training_time, predictions_preview
        """
        start = _time.time()
        tmpl = MODEL_TEMPLATES.get(model_name)
        if not tmpl:
            return json.dumps({"error": f"Unknown model: {model_name}"})

        # Check multivariate compatibility
        if bool(feat_cols) and X is not None:
            if not tmpl["supports_multivariate"]:
                return json.dumps({
                    "error": f"{model_name} does not support multivariate forecasting. "
                            f"Use RandomForest or LightGBM for multivariate."
                })

        is_mv = (
            bool(feat_cols)
            and tmpl["supports_multivariate"]
            and X is not None
            and model_name in mv_tools
        )

        combos = _param_combinations(tmpl["search_space"]) if tmpl["search_space"] else [tmpl["hyperparameters"]]

        best_score, best_params, best_cv = float("inf"), tmpl["hyperparameters"], []
        for params in combos:
            if is_mv:
                scores = _cv_multivariate(mv_tools[model_name], y_mv, X, 3, horizon, params)
            else:
                scores = _cv_univariate(uni_tools[model_name], y, 3, horizon, params)
            finite = [s for s in scores if np.isfinite(s)]
            ms = np.mean(finite) if finite else float("inf")
            if ms < best_score:
                best_score, best_params, best_cv = ms, params, scores

        # Final predictions on full data
        if is_mv:
            test_X = X[-horizon:] if len(X) >= horizon else X
            rj = mv_tools[model_name].invoke({
                "train_features": json.dumps(X.tolist()),
                "train_target": json.dumps(y_mv.tolist()),
                "test_features": json.dumps(test_X.tolist()),
                "horizon": horizon,
                **{k: v for k, v in best_params.items() if k != "n_lags"},
            })
        else:
            rj = uni_tools[model_name].invoke({
                "train_data": json.dumps(y.tolist()),
                "horizon": horizon, **best_params,
            })

        final = json.loads(rj)
        predictions = final["predictions"]
        finite_best = [s for s in best_cv if np.isfinite(s)]
        mean_mape = float(np.mean(finite_best)) if finite_best else float("inf")
        std_mape = float(np.std(finite_best)) if len(finite_best) > 1 else 0.0
        tt = _time.time() - start

        warn = ""
        if model_name == "ARIMA":
            # Use actual fitted params from metadata, not input params
            meta = final.get("metadata", {})
            p_fit = meta.get("p", best_params.get("p", 0))
            q_fit = meta.get("q", best_params.get("q", 0))
            d_fit = meta.get("d", best_params.get("d", 0))
            if p_fit == 0 and q_fit == 0:
                warn = f"WARNING: ARIMA(0,{d_fit},0) is a random walk"

        session["model_results"][model_name] = {
            "predictions": predictions,
            "cv_scores": best_cv,
            "mean_mape": mean_mape,
            "std_mape": std_mape,
            "best_params": best_params,
            "training_time": tt,
            "metadata": final.get("metadata", {}),
            "is_multivariate": is_mv,
        }

        out = {
            "model_name": model_name,
            "mean_mape": round(mean_mape, 4),
            "std_mape": round(std_mape, 4),
            "best_params": best_params,
            "training_time": round(tt, 1),
            "predictions_preview": [round(v, 2) for v in predictions[:5]],
            "n_predictions": len(predictions),
            "multivariate": is_mv,
        }
        if warn:
            out["warning"] = warn
        return json.dumps(out)

    # ===== TOOL 3: Train + Explain =====
    @tool
    def train_and_explain_forecast(model_name: str, horizon: int) -> str:
        """Train a model AND compute explainability (SHAP, feature importance,
        STL decomposition, lag correlations).

        Use when the user asks for explanation. Best with tree models
        (RandomForest, LightGBM) which support SHAP/FI.

        Args:
            model_name: Choose from available models.
                UNIVARIATE: Naive|SeasonalNaive|ARIMA|ETS|N-BEATS|TinyTimeMixer
                MULTIVARIATE (requires features): RandomForest|LightGBM
            horizon: Number of time steps to forecast

        Returns:
            JSON with forecast results + explainability metrics
        """
        from tinyts.tools.explainability import (
            compute_statistical_summary, compute_decomposition_metrics,
            compute_lag_contributions, compute_feature_importance,
            compute_shap_values, compute_feature_correlations,
        )

        # Train first
        fc_json = train_forecast_model.invoke({"model_name": model_name, "horizon": horizon})
        fc = json.loads(fc_json)
        if "error" in fc:
            return json.dumps(fc)

        stored = session["model_results"].get(model_name, {})
        bp = stored.get("best_params", {})
        is_mv = stored.get("is_multivariate", False)

        # --- Shared metrics: compute ONCE, cache in session ---
        if "_shared_explain" not in session:
            series = pd.Series(y)
            sp = 24
            if session.get("profile") and session["profile"].seasonal_period:
                sp = session["profile"].seasonal_period
            n = len(series)
            split = int(n * 0.8)
            train_s, test_s = series.iloc[:split], series.iloc[split:]

            stat = compute_statistical_summary(train_s, test_s, periods_per_day=sp)
            decomp = compute_decomposition_metrics(train_s, seasonal_period=sp)
            lags = compute_lag_contributions(train_s)
            corrs = {}
            if is_mv and X is not None and feat_cols:
                X_df = pd.DataFrame(X, columns=feat_cols)
                corrs = compute_feature_correlations(X_df, pd.Series(y_mv), feat_cols)

            # Round shared metrics to 3 decimals
            stat = {k: round(v, 3) if isinstance(v, float) else v for k, v in stat.items()}
            decomp = {k: round(v, 3) if isinstance(v, float) else v for k, v in decomp.items()}
            lags = {k: {ik: round(iv, 3) if isinstance(iv, float) else iv for ik, iv in v.items()}
                    for k, v in lags.items()}
            corrs = {k: round(v, 3) for k, v in corrs.items()}

            # Lag interpretation
            lag_hints = []
            lag1 = lags.get("lag_1", {}).get("correlation", 0)
            if lag1 < -0.1:
                lag_hints.append(f"lag_1={lag1}: mean-reverting")
            elif lag1 > 0.3:
                lag_hints.append(f"lag_1={lag1}: persistent/momentum")
            else:
                lag_hints.append(f"lag_1={lag1}: weak short-term dependence")
            for lk in ["lag_12", "lag_24"]:
                lv = lags.get(lk, {}).get("correlation", 0)
                if abs(lv) < 0.1:
                    lag_hints.append(f"{lk}={lv}: weak seasonality")
                elif lv > 0.3:
                    lag_hints.append(f"{lk}={lv}: strong seasonal signal")

            session["_shared_explain"] = {
                "stat_summary": stat,
                "decomposition": decomp,
                "lags": {k: round(v.get("correlation", 0), 3) for k, v in lags.items()},
                "lag_hints": lag_hints,
                "correlations": corrs,
            }

        shared = session["_shared_explain"]

        # --- Per-model metrics: FI + SHAP ---
        fi, shap_v = {}, {}

        if is_mv and X is not None and feat_cols:
            X_df = pd.DataFrame(X, columns=feat_cols)
            if model_name == "RandomForest":
                from sklearn.ensemble import RandomForestRegressor
                mdl = RandomForestRegressor(
                    n_estimators=bp.get("n_estimators", 100),
                    max_depth=bp.get("max_depth", 10),
                    random_state=42, n_jobs=-1,
                )
                mdl.fit(X_df, y_mv)
                fi = compute_feature_importance(mdl, feat_cols, model_name)
                shap_v = compute_shap_values(mdl, X_df.iloc[-min(10, len(X_df)):], feat_cols, model_name)
            elif model_name == "LightGBM":
                import lightgbm as lgb
                mdl = lgb.LGBMRegressor(
                    num_leaves=bp.get("num_leaves", 31),
                    learning_rate=bp.get("learning_rate", 0.1),
                    n_estimators=bp.get("n_estimators", 100),
                    random_state=42, verbose=-1,
                )
                mdl.fit(X_df, y_mv)
                fi = compute_feature_importance(mdl, feat_cols, model_name)
                shap_v = compute_shap_values(mdl, X_df.iloc[-min(10, len(X_df)):], feat_cols, model_name)

        elif model_name in ("RandomForest", "LightGBM"):
            from tinyts.tools.tree_based import create_lagged_features
            nl = bp.get("n_lags", 12)
            Xl, yl = create_lagged_features(y, nl)
            lag_names = [f"lag_{i+1}" for i in range(nl)]
            Xl_df = pd.DataFrame(Xl, columns=lag_names)
            if model_name == "RandomForest":
                from sklearn.ensemble import RandomForestRegressor
                mdl = RandomForestRegressor(
                    n_estimators=bp.get("n_estimators", 100),
                    max_depth=bp.get("max_depth", 10),
                    random_state=42, n_jobs=-1,
                )
            else:
                import lightgbm as lgb
                mdl = lgb.LGBMRegressor(
                    num_leaves=bp.get("num_leaves", 31),
                    learning_rate=bp.get("learning_rate", 0.1),
                    n_estimators=bp.get("n_estimators", 100),
                    random_state=42, verbose=-1,
                )
            mdl.fit(Xl_df, yl)
            fi = compute_feature_importance(mdl, lag_names, model_name)
            shap_v = compute_shap_values(mdl, Xl_df.iloc[-min(10, len(Xl_df)):], lag_names, model_name)

        # Normalize FI to percentages (sum=100%)
        fi_total = sum(abs(v) for v in fi.values()) if fi else 1
        fi_pct = {k: round(abs(v) / fi_total * 100, 1) for k, v in fi.items()} if fi else {}

        # Normalize SHAP to relative percentages (sum=100%)
        shap_total = sum(abs(v) for v in shap_v.values()) if shap_v else 1
        shap_pct = {k: round(abs(v) / shap_total * 100, 1) for k, v in shap_v.items()} if shap_v else {}

        session["model_explanations"][model_name] = {
            "feature_importance_pct": fi_pct,
            "shap_pct": shap_pct,
        }

        # --- Reconciliation hints ---
        corrs = shared["correlations"]
        reconciliation = []
        for feat in set(list(fi_pct.keys()) + list(shap_pct.keys()) + list(corrs.keys())):
            f_val = fi_pct.get(feat, 0)
            s_val = shap_pct.get(feat, 0)
            c_val = corrs.get(feat, 0)
            hint = f"{feat}: FI={f_val}%, SHAP={s_val}%, corr={c_val}"
            if s_val > 10 and abs(c_val) > 0.2:
                hint += " → direct driver"
            elif abs(c_val) < 0.1 and f_val > 5:
                hint += " → non-linear effect"
            reconciliation.append(hint)

        # Compact output — shared metrics only on FIRST model
        explain_out = {
            "fi_pct": fi_pct,
            "shap_pct": shap_pct,
            "reconciliation": reconciliation,
        }
        if len(session["model_explanations"]) == 1:
            explain_out["data_profile"] = {
                "stat_summary": shared["stat_summary"],
                "decomposition": shared["decomposition"],
                "lags": shared["lags"],
                "lag_hints": shared["lag_hints"],
                "correlations": shared["correlations"],
            }

        fc["explainability"] = explain_out
        print(fc)
        return json.dumps(fc)

    # ===== TOOL 4: Ensemble Strategy =====
    @tool
    def select_ensemble_strategy() -> str:
        """Select ensemble strategy from all trained models.

        Call AFTER training all desired models. Computes inverse-error
        weights from CV scores. Takes no arguments.

        Returns:
            JSON with strategy_type, model_weights, selected_models
        """
        from tinyts.tools.ensemble import compute_ensemble_weights as _cew

        results = session["model_results"]
        if not results:
            return json.dumps({"error": "No models trained yet."})

        if len(results) == 1:
            name = list(results.keys())[0]
            r = results[name]
            strat = {
                "strategy_type": "best_model",
                "model_weights": {name: 1.0},
                "selected_models": [name],
                "justification": f"Only one model trained ({name}, MAPE={r['mean_mape']:.2f}%).",
            }
            session["ensemble_strategy"] = strat
            return json.dumps(strat)

        top = sorted(results.items(), key=lambda x: x[1]["mean_mape"])[:5]
        scores = {n: d["mean_mape"] for n, d in top}
        stds = {n: d["std_mape"] for n, d in top}
        wj = json.loads(_cew.invoke({
            "model_scores": json.dumps(scores), "method": "inverse_error",
        }))
        weights = wj["weights"]

        # Build justification
        families = {n: MODEL_TEMPLATES.get(n, {}).get("family", "?") for n, _ in top}
        unique_families = set(families.values())
        best_name, best_score = top[0]
        worst_name, worst_score = top[-1]
        spread = worst_score[1]["mean_mape"] - best_score[1]["mean_mape"] if len(top) > 1 else 0

        justification_parts = [
            f"Best model: {top[0][0]} (MAPE={top[0][1]['mean_mape']:.2f}%±{top[0][1]['std_mape']:.2f}%)",
            f"Model families: {', '.join(f'{n}={families[n]}' for n, _ in top)}",
            f"Diversity: {len(unique_families)} family types ({', '.join(unique_families)})",
            f"Error spread: {top[-1][1]['mean_mape']:.2f}% - {top[0][1]['mean_mape']:.2f}% = {top[-1][1]['mean_mape'] - top[0][1]['mean_mape']:.2f}pp",
        ]
        avg_std = np.mean([d["std_mape"] for _, d in top])
        if avg_std > 5:
            justification_parts.append(f"High CV variance (avg std={avg_std:.1f}%) → weighting helps stabilize")
        else:
            justification_parts.append(f"Low CV variance (avg std={avg_std:.1f}%) → models fairly stable")

        strat = {
            "strategy_type": "weighted",
            "model_weights": weights,
            "selected_models": list(weights.keys()),
            "per_model_scores": {n: {"mape": round(d["mean_mape"], 2), "std": round(d["std_mape"], 2)} for n, d in top},
            "justification": justification_parts,
        }
        session["ensemble_strategy"] = strat
        return json.dumps(strat)

    # ===== TOOL 5: Combine Forecasts =====
    @tool
    def combine_forecasts() -> str:
        """Combine predictions using inverse-MAPE weighting.

        Call AFTER training all models. Automatically computes weights
        from CV scores and generates final forecast. Takes no arguments.

        Returns:
            JSON with n_predictions, preview, strategy, models_used
        """
        from tinyts.tools.ensemble import combine_predictions as _cp

        results = session["model_results"]
        if not results:
            return json.dumps({"error": "No models trained yet."})

        # Auto-compute inverse-MAPE weights
        top = sorted(results.items(), key=lambda x: x[1]["mean_mape"])[:5]
        scores = {n: d["mean_mape"] for n, d in top}

        if len(top) == 1:
            strat = {
                "strategy_type": "best_model",
                "model_weights": {top[0][0]: 1.0},
                "selected_models": [top[0][0]],
            }
            preds = results[top[0][0]]["predictions"]
        else:
            # Inverse-MAPE weighting
            inv_scores = {n: 1.0 / max(s, 0.01) for n, s in scores.items()}
            total = sum(inv_scores.values())
            weights = {n: v / total for n, v in inv_scores.items()}

            strat = {
                "strategy_type": "weighted",
                "model_weights": weights,
                "selected_models": list(weights.keys()),
            }

            pd_dict = {n: results[n]["predictions"] for n in weights.keys()}
            cj = json.loads(_cp.invoke({
                "predictions_dict": json.dumps(pd_dict),
                "weights_dict": json.dumps(weights),
            }))
            preds = cj["combined_predictions"]

        session["ensemble_strategy"] = strat
        session["final_predictions"] = preds
        _save_forecast_plot(y, preds, results, strat["selected_models"], output_dir)

        return json.dumps({
            "n_predictions": len(preds),
            "predictions_preview": [round(v, 2) for v in preds[:10]],
            "strategy": strat["strategy_type"],
            "models_used": strat["selected_models"],
            "weights": strat["model_weights"],
        })

    # ===== TOOL 6: Detect Anomalies =====
    @tool
    def detect_anomalies(confirm: str = "yes") -> str:
        """Run 7-method anomaly ensemble (Z-score, MAD, Rolling, IQR,
        STL, IsolationForest, DBSCAN) with majority voting.

        Args:
            confirm: Just pass "yes" to confirm.

        Returns:
            JSON with n_anomalies, per-method counts, average_agreement
        """
        from tinyts.tools.anomaly import run_anomaly_ensemble as _rae
        result = json.loads(_rae.invoke({"data": json.dumps(y.tolist())}))
        session["anomaly_results"] = result
        _save_anomaly_plot(y, result, target_column, output_dir)

        return json.dumps({
            "n_anomalies": result["n_anomalies"],
            "method_counts": result["method_counts"],
            "min_votes": result["metadata"]["min_votes"],
            "average_agreement": round(result["metadata"]["average_agreement"], 1),
            "total_data_points": len(y),
        })

    # ===== TOOL 7: Explain Anomalies =====
    @tool
    def explain_anomalies(confirm: str = "yes") -> str:
        """Compute explanation metrics for detected anomalies.

        Call AFTER detect_anomalies(). Returns z-score context, IQR bounds,
        STL residuals, method agreement, percentile context.

        Returns:
            JSON with explanation metrics
        """
        from tinyts.tools.explainability import (
            compute_zscore_explanation, compute_iqr_explanation,
            compute_stl_anomaly_explanation, compute_method_agreement,
            compute_percentile_context,
        )
        anom = session.get("anomaly_results")
        if not anom:
            return json.dumps({"error": "Call detect_anomalies first."})

        series = pd.Series(y)
        sp = 24
        if session.get("profile") and session["profile"].seasonal_period:
            sp = session["profile"].seasonal_period

        labels = np.array(anom["anomaly_labels"])
        a_idx = pd.Index(series.index[labels == -1]) if len(labels) == len(series) else pd.Index([])
        a_vals = series.iloc[labels == -1].tolist() if len(labels) == len(series) else []

        exp = {
            "zscore": compute_zscore_explanation(series, a_idx),
            "iqr": compute_iqr_explanation(series),
            "stl": compute_stl_anomaly_explanation(series, seasonal_period=sp),
            "method_agreement": compute_method_agreement(anom["method_counts"], 7),
            "percentiles": compute_percentile_context(series, a_vals) if a_vals else {},
        }
        session["anomaly_explanation"] = exp
        return json.dumps(exp, default=str)

    # ===== TOOL 8: Generate Report =====
    @tool
    def generate_report(confirm: str = "yes") -> str:
        """Generate a natural-language analysis report.

        Call LAST after all analysis. Uses LLM to synthesize results.

        Returns:
            The full report text (markdown)
        """
        from tinyts.config import get_llm

        parts = []
        prof = session.get("profile")
        if prof:
            parts.append(
                f"DATASET: {prof.shape[0]} rows, freq={prof.inferred_frequency}, "
                f"seasonality={prof.seasonal_period}, trend={prof.has_trend}"
            )

        res = session.get("model_results")
        if res:
            lines = [f"  {n}: MAPE={d['mean_mape']:.2f}% (+/-{d['std_mape']:.2f}%)"
                     for n, d in sorted(res.items(), key=lambda x: x[1]["mean_mape"])]
            parts.append("MODEL RESULTS:\n" + "\n".join(lines))

        strat = session.get("ensemble_strategy")
        if strat:
            parts.append(f"STRATEGY: {strat['strategy_type']} — {strat['selected_models']}")

        anom = session.get("anomaly_results")
        if anom:
            parts.append(f"ANOMALIES: {anom['n_anomalies']} found, counts={anom['method_counts']}")

        for name, exp in session.get("model_explanations", {}).items():
            fi = exp.get("feature_importance", {})
            if fi:
                top = sorted(fi.items(), key=lambda x: -abs(x[1]))[:5]
                parts.append(f"FEATURE IMPORTANCE ({name}): {top}")

        aexp = session.get("anomaly_explanation")
        if aexp:
            parts.append(f"ANOMALY EXPLANATION: {aexp.get('method_agreement', {})}")

        ctx = "\n\n".join(parts)
        is_anomaly_only = anom and not res

        if is_anomaly_only:
            prompt = (
                "You are a data science report writer. Generate a CONCISE scientific "
                "report for anomaly detection.\n\n"
                f"{ctx}\n\n"
                "Write 300-500 words with sections: Dataset, Methodology, Results, "
                "Explainability, Limitations, Recommendations.\n\nReport:"
            )
        else:
            prompt = (
                "You are a data science report writer. Generate a CONCISE scientific report.\n\n"
                f"{ctx}\n\n"
                "Write 300-500 words with sections: Dataset, Approach, Results, Strategy, "
                "Explainability, Limitations, Recommendations.\n\nReport:"
            )

        try:
            llm = get_llm(temperature=0.3)
            report = llm.invoke(prompt).content
        except Exception as e:
            report = f"Report generation failed: {e}\n\n{ctx}"

        rp = Path(output_dir) / "final_report.md"
        rp.write_text("# Time Series Analysis Report\n\n" + report)
        session["report"] = report
        return report

    return [
        profile_dataset,
        train_forecast_model,
        train_and_explain_forecast,
        select_ensemble_strategy,
        combine_forecasts,
        detect_anomalies,
        explain_anomalies,
        generate_report,
    ], session
