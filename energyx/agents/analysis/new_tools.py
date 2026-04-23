"""Analysis Agent tools — 14-tool redesign.

Internal tinyts tools (train_forecast_model, combine_forecasts, etc.) are passed
in via `internal_tools` and called directly.  They are NOT exposed to the LLM.
Only the 14 high-level tools here are visible to the agent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_UNIT_RATE = 0.2459   # £/kWh fallback
_STANDING   = 0.61    # £/day fallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tariff():
    try:
        from energyx.data.tariffs import get_tariff_cache
        return get_tariff_cache().get_active_tariff()
    except Exception:
        return None


def _tariff_rate(tariff) -> float:
    try:
        return tariff.rates[0].unit_rate_gbp_per_kwh
    except Exception:
        return _UNIT_RATE


def _freq_to_seconds(freq_str: Optional[str]) -> float:
    if not freq_str:
        return 60.0
    import re
    freq_str = freq_str.upper().strip()
    mapping = {"S": 1, "T": 60, "MIN": 60, "H": 3600, "D": 86400, "W": 604800}
    m = re.match(r"^(\d*)([A-Z]+)$", freq_str)
    if m:
        n = int(m.group(1)) if m.group(1) else 1
        unit = m.group(2)
        return float(n * mapping.get(unit, 60))
    return 60.0


def _parse_model_list(models_str: Optional[str], univariate: bool = True) -> List[str]:
    UNI  = ["Naive", "SeasonalNaive", "ARIMA", "ETS", "N-BEATS"]
    MULTI = ["RandomForest", "LightGBM"]
    if not models_str or models_str.strip().lower() in ("all", ""):
        return UNI if univariate else MULTI
    raw = [m.strip() for m in models_str.replace(";", ",").split(",") if m.strip()]
    canonical = {
        "naive": "Naive", "seasonalnaive": "SeasonalNaive",
        "seasonal_naive": "SeasonalNaive", "arima": "ARIMA",
        "ets": "ETS", "nbeats": "N-BEATS", "n_beats": "N-BEATS",
        "n-beats": "N-BEATS", "randomforest": "RandomForest",
        "random_forest": "RandomForest", "lightgbm": "LightGBM", "lgbm": "LightGBM",
    }
    pool = UNI if univariate else MULTI
    result = []
    for r in raw:
        c = canonical.get(r.lower(), r)
        if c in pool:
            result.append(c)
    return result or pool


def _kwh_from_watts(watts_mean: float, step_sec: float, n_steps: int) -> float:
    return watts_mean / 1000.0 * (step_sec / 3600) * n_steps


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def create_analysis_extensions(
    session: Dict[str, Any],
    dataset_path: str,
    time_column: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    internal_tools: Optional[Dict[str, Any]] = None,
) -> List:
    """Return the 14 high-level LangChain tools bound to this session."""

    internal = internal_tools or {}
    feat_cols = feature_columns or []

    def _call(name: str, args: dict) -> dict:
        """Call an internal tinyts tool by name; return parsed dict."""
        fn = internal.get(name)
        if fn is None:
            return {"error": f"internal tool '{name}' not available"}
        try:
            return json.loads(fn.invoke(args))
        except Exception as e:
            return {"error": str(e)}

    # ===========================================================
    # 1 — forecast_univariate
    # ===========================================================
    @tool
    def forecast_univariate(horizon: int = 24, models: str = "all") -> str:
        """Run all univariate forecasting models, pick best by MAPE, save forecast plot.

        Args:
            horizon: Steps to forecast (default 24).
            models:  Comma-separated model names or "all".
                     Available: Naive, SeasonalNaive, ARIMA, ETS, N-BEATS
        Returns JSON with best_model, predictions, model_comparison table.
        """
        try:
            horizon = int(horizon)
            model_list = _parse_model_list(models, univariate=True)
            all_res: Dict[str, dict] = {}
            for mn in model_list:
                r = _call("train_forecast_model", {"model_name": mn, "horizon": horizon})
                if "error" not in r:
                    all_res[mn] = r

            if not all_res:
                return json.dumps({"error": "All univariate models failed. Check data quality."})

            best_name = min(all_res, key=lambda n: all_res[n].get("mean_mape", float("inf")))
            best_r = all_res[best_name]
            preds = session.get("model_results", {}).get(best_name, {}).get("predictions", [])
            session["final_predictions"] = preds

            comparison = {
                n: {"mape": round(r.get("mean_mape", 0), 3),
                    "std":  round(r.get("std_mape", 0), 3),
                    "multivariate": r.get("multivariate", False)}
                for n, r in all_res.items()
            }

            # Save forecast plot
            try:
                from tinyts.agent_tools import _save_forecast_plot
                import numpy as np
                df_plot = pd.read_csv(dataset_path)
                y_plot = df_plot[target_column].dropna().values.astype(float)
                _save_forecast_plot(y_plot, preds, session.get("model_results", {}),
                                    best_name, session.get("output_dir", "."))
            except Exception:
                pass

            return json.dumps({
                "status": "ok",
                "best_model": best_name,
                "best_mape_pct": round(best_r.get("mean_mape", 0), 3),
                "horizon": horizon,
                "predictions": [round(v, 2) for v in preds],
                "model_comparison": comparison,
            })
        except Exception as e:
            return json.dumps({"error": f"forecast_univariate failed: {e}"})

    # ===========================================================
    # 2 — forecast_multivariate
    # ===========================================================
    @tool
    def forecast_multivariate(horizon: int = 24, models: str = "all") -> str:
        """Run multivariate forecasting models (RandomForest, LightGBM) using available
        feature columns. Picks best by MAPE. Falls back to univariate if no features loaded.

        Args:
            horizon: Steps to forecast (default 24).
            models:  "all", "RandomForest", "LightGBM", or comma-separated.
        """
        try:
            horizon = int(horizon)
            if not feat_cols:
                return json.dumps({
                    "error": "No feature columns available. Use forecast_univariate instead.",
                    "available_features": feat_cols,
                })

            model_list = _parse_model_list(models, univariate=False)
            all_res: Dict[str, dict] = {}
            for mn in model_list:
                r = _call("train_forecast_model", {"model_name": mn, "horizon": horizon})
                if "error" not in r:
                    all_res[mn] = r

            if not all_res:
                return json.dumps({"error": "All multivariate models failed."})

            best_name = min(all_res, key=lambda n: all_res[n].get("mean_mape", float("inf")))
            best_r = all_res[best_name]
            preds = session.get("model_results", {}).get(best_name, {}).get("predictions", [])
            session["final_predictions"] = preds

            comparison = {
                n: {"mape": round(r.get("mean_mape", 0), 3),
                    "std":  round(r.get("std_mape", 0), 3),
                    "multivariate": r.get("multivariate", False)}
                for n, r in all_res.items()
            }

            # Save forecast plot
            try:
                from tinyts.agent_tools import _save_forecast_plot
                import numpy as np
                df_plot = pd.read_csv(dataset_path)
                y_plot = df_plot[target_column].dropna().values.astype(float)
                _save_forecast_plot(y_plot, preds, session.get("model_results", {}),
                                    best_name, session.get("output_dir", "."))
            except Exception:
                pass

            return json.dumps({
                "status": "ok",
                "best_model": best_name,
                "best_mape_pct": round(best_r.get("mean_mape", 0), 3),
                "horizon": horizon,
                "feature_columns": feat_cols,
                "predictions": [round(v, 2) for v in preds],
                "model_comparison": comparison,
            })
        except Exception as e:
            return json.dumps({"error": f"forecast_multivariate failed: {e}"})

    # ===========================================================
    # 3 — explain_forecast
    # ===========================================================
    @tool
    def explain_forecast(model_name: str = "best") -> str:
        """Full causal explainability for a forecast: SHAP, feature importance,
        STL decomposition, lag autocorrelations, weather/feature correlations,
        and consumption attribution (weather vs behavioural vs appliance).

        Call AFTER forecast_univariate or forecast_multivariate.

        Args:
            model_name: Model to explain, or "best" to auto-select from session.
        """
        try:
            from tinyts.tools.explainability import (
                compute_statistical_summary, compute_decomposition_metrics,
                compute_lag_contributions, compute_feature_importance,
                compute_shap_values, compute_feature_correlations,
            )

            # Resolve model
            results = session.get("model_results", {})
            if not results:
                return json.dumps({"error": "Run forecast_univariate or forecast_multivariate first."})

            if model_name == "best" or model_name not in results:
                model_name = min(results, key=lambda n: results[n].get("mean_mape", float("inf")))

            r = _call("train_and_explain_forecast",
                      {"model_name": model_name, "horizon": results[model_name].get("horizon", 24)
                       or len(results[model_name].get("predictions", [24]))})

            # Shared explain cache already built by train_and_explain_forecast
            shared = session.get("_shared_explain", {})
            exp = r.get("explainability", {})
            fi_pct  = exp.get("fi_pct", {})
            shap_pct = exp.get("shap_pct", {})

            # Attribution buckets (weather / appliance / room / behavioural)
            weather_kw  = {"temperature", "temp", "humidity", "wind", "weather", "gas_watts",
                           "temperature_mean", "humidity_mean", "weather_temp_c",
                           "weather_humidity_pct"}
            appliance_kw = {"appl_", "appliance", "circuit_", "gas"}
            room_kw      = {"temp_room", "room"}
            occupancy_kw = {"light_mean", "light", "occupancy", "presence"}

            def _bucket(fi: dict) -> dict:
                total = sum(fi.values()) or 1
                def _sum(kws):
                    return sum(v for k, v in fi.items() if any(w in k.lower() for w in kws))
                w = _sum(weather_kw)
                a = _sum(appliance_kw)
                rm = _sum(room_kw)
                oc = _sum(occupancy_kw)
                beh = max(0, total - w - a - rm - oc)
                return {
                    "weather_pct":    round(w / total * 100, 1),
                    "appliance_pct":  round(a / total * 100, 1),
                    "room_pct":       round(rm / total * 100, 1),
                    "occupancy_pct":  round(oc / total * 100, 1),
                    "behavioural_pct": round(beh / total * 100, 1),
                }

            attribution = _bucket(fi_pct) if fi_pct else _bucket(shap_pct)

            return json.dumps({
                "status": "ok",
                "model": model_name,
                "mape_pct": round(results[model_name].get("mean_mape", 0), 3),
                "stat_summary": shared.get("stat_summary", {}),
                "decomposition": shared.get("decomposition", {}),
                "lags": shared.get("lags", {}),
                "lag_hints": shared.get("lag_hints", []),
                "feature_correlations": shared.get("correlations", {}),
                "feature_importance_pct": fi_pct,
                "shap_pct": shap_pct,
                "reconciliation": exp.get("reconciliation", []),
                "attribution": attribution,
                "interpretation": (
                    "SHAP values confirm causal drivers." if shap_pct else
                    "Feature importance used (SHAP unavailable for this model)."
                ),
            })
        except Exception as e:
            return json.dumps({"error": f"explain_forecast failed: {e}"})

    # ===========================================================
    # 4 — detect_anomalies
    # ===========================================================
    @tool
    def detect_anomalies(methods: str = "all", level: str = "home") -> str:
        """Run anomaly detection ensemble on energy consumption.

        Args:
            methods: Comma-separated subset of: zscore, mad, rolling, iqr, stl,
                     isolationforest, dbscan — or "all" (default).
            level:   "home" (electricity target) or "appliance" (all appl_* columns).
        Returns JSON with n_anomalies, per-method counts, anomaly_labels, average_agreement.
        """
        try:
            r = _call("detect_anomalies", {"confirm": "yes"})
            if "error" in r:
                return json.dumps(r)

            # If methods != "all", filter method_counts to only show requested
            method_counts = r.get("method_counts", {})
            if methods.strip().lower() != "all":
                requested = {m.strip().lower() for m in methods.split(",")}
                method_counts = {k: v for k, v in method_counts.items()
                                 if k.lower() in requested}

            # Appliance-level anomalies: check appl_* columns for unusual events
            appl_anomalies: Dict[str, Any] = {}
            if level == "appliance":
                try:
                    df_csv = pd.read_csv(dataset_path, nrows=50000)
                    appl_cols = [c for c in df_csv.columns if c.startswith("appl_")]
                    for col in appl_cols:
                        vals = pd.to_numeric(df_csv[col], errors="coerce").dropna()
                        if len(vals) < 20:
                            continue
                        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                        iqr = q3 - q1
                        upper = q3 + 3 * iqr
                        n_spikes = int((vals > upper).sum())
                        if n_spikes > 0:
                            appl_anomalies[col.replace("appl_", "")] = {
                                "n_spikes": n_spikes,
                                "threshold_watts": round(float(upper), 1),
                                "max_watts": round(float(vals.max()), 1),
                            }
                except Exception:
                    pass

            return json.dumps({
                "status": r.get("status", "ok"),
                "level": level,
                "n_anomalies": r.get("n_anomalies", 0),
                "total_data_points": r.get("total_data_points", 0),
                "method_counts": method_counts,
                "average_agreement": r.get("average_agreement", 0),
                "min_votes": r.get("min_votes", 2),
                "multivariate": r.get("multivariate", False),
                "appliance_anomalies": appl_anomalies,
                "warnings": r.get("warnings", []),
            })
        except Exception as e:
            return json.dumps({"error": f"detect_anomalies failed: {e}"})

    # ===========================================================
    # 5 — explain_anomalies
    # ===========================================================
    @tool
    def explain_anomalies() -> str:
        """Explain detected anomalies: z-score context, IQR bounds, STL residuals,
        method agreement, percentile ranking, context windows.

        Call AFTER detect_anomalies(). Uses explainability.py functions directly.
        """
        try:
            r = _call("explain_anomalies", {"confirm": "yes"})
            if "error" in r:
                return json.dumps(r)

            # Build user-friendly narrative from the explain metrics
            zscore = r.get("zscore", {})
            iqr    = r.get("iqr", {})
            stl    = r.get("stl", {})
            agree  = r.get("method_agreement", {})
            pcts   = r.get("percentiles", {})
            snaps  = r.get("context_snapshots", [])

            n_anom = len(snaps)
            mean_v = zscore.get("mean", 0)
            std_v  = zscore.get("std", 1)
            iqr_lo = iqr.get("lower_bound", mean_v - 3 * std_v)
            iqr_hi = iqr.get("upper_bound", mean_v + 3 * std_v)
            stl_thresh = stl.get("anomaly_threshold", 3 * stl.get("residual_std", std_v))

            samples = []
            for s in zscore.get("sample_explanations", [])[:5]:
                val = s["value"]
                z   = s["zscore"]
                pct = pcts.get("anomaly_percentiles", {}).get(str(val), 0)
                kind = "spike" if val > mean_v else "drop"
                interp = ("appliance switched on / demand surge" if kind == "spike"
                          else "outage / sensor dropout / appliance off")
                samples.append({
                    "timestamp":   s["timestamp"],
                    "value":       round(val, 1),
                    "z_score":     round(z, 2),
                    "percentile":  round(pct, 1),
                    "type":        kind,
                    "likely_cause": interp,
                })

            return json.dumps({
                "status": "ok",
                "n_anomalies": agree.get("method_counts") and sum(1 for v in agree.get("method_counts", {}).values() if v > 0),
                "normal_range_watts": f"{iqr_lo:.0f}–{iqr_hi:.0f}",
                "stl_residual_std":   round(stl.get("residual_std", 0), 2),
                "stl_threshold_3sigma": round(stl_thresh, 2),
                "average_method_agreement": round(agree.get("average_agreement", 0), 2),
                "anomaly_samples":  samples,
                "context_windows":  snaps[:3],
                "p95_watts":  pcts.get("p95", 0),
                "p99_watts":  pcts.get("p99", 0),
                "multivariate": r.get("multivariate", False),
            })
        except Exception as e:
            return json.dumps({"error": f"explain_anomalies failed: {e}"})

    # ===========================================================
    # 6 — counterfactual_fwd
    # ===========================================================
    @tool
    def counterfactual_fwd(
        changes_json: str,
        horizon: int = 24,
        level: str = "consumption",
    ) -> str:
        """Forward what-if: estimate impact of feature/appliance/billing changes.

        Args:
            changes_json: JSON dict of feature deltas.
                consumption/appliance level: fractional changes, e.g. '{"heating": 0.2}'
                bill level uses same changes but outputs £ impact.
            horizon: Steps to project (default 24).
            level:   "consumption" (watts), "appliance" (appl-level delta),
                     or "bill" (applies active tariff to get £ impact).
        """
        try:
            try:
                changes = json.loads(changes_json)
            except Exception:
                return json.dumps({"error": f"changes_json must be valid JSON: {changes_json!r}"})

            tariff = _get_tariff()
            rate = _tariff_rate(tariff)

            if level == "consumption" and feat_cols:
                # Full multivariate counterfactual
                r = _call("counterfactual_forward",
                          {"changes_json": changes_json, "horizon": horizon})
                if "error" in r:
                    return json.dumps(r)
                out = {
                    "status": "ok", "level": "consumption",
                    "baseline_mean_watts":       r.get("baseline_mean"),
                    "counterfactual_mean_watts": r.get("counterfactual_mean"),
                    "mean_impact_watts":         r.get("mean_impact"),
                    "feature_changes":           r.get("feature_changes"),
                    "model_used":                r.get("model"),
                }
                return json.dumps(out)

            # Proportional estimate (no multivariate model or bill level)
            profile = session.get("profile")
            freq_str = getattr(profile, "inferred_frequency", None) if profile else None
            step_sec = _freq_to_seconds(freq_str)
            total_hours = horizon * step_sec / 3600

            baseline_mean = getattr(profile, "mean", 500.0) or 500.0
            final_preds = session.get("final_predictions", [])
            if final_preds:
                baseline_mean = float(np.mean(final_preds))

            net_scale = 1.0 + sum(float(v) for v in changes.values())
            cf_mean = baseline_mean * max(0, net_scale)

            baseline_kwh = _kwh_from_watts(baseline_mean, step_sec, horizon)
            cf_kwh       = _kwh_from_watts(cf_mean, step_sec, horizon)

            out: Dict[str, Any] = {
                "status": "ok",
                "level": level,
                "method": "proportional_scaling",
                "horizon_hours": round(total_hours, 1),
                "baseline_mean_watts":       round(baseline_mean, 1),
                "counterfactual_mean_watts": round(cf_mean, 1),
                "mean_impact_watts":         round(cf_mean - baseline_mean, 1),
                "changes_applied": changes,
            }

            if level in ("bill", "appliance"):
                standing = (tariff.standing_charge_gbp_per_day * total_hours / 24
                            if tariff else _STANDING * total_hours / 24)
                base_bill = baseline_kwh * rate + standing
                cf_bill   = cf_kwh   * rate + standing
                out.update({
                    "baseline_bill_gbp":       round(base_bill, 4),
                    "counterfactual_bill_gbp": round(cf_bill, 4),
                    "bill_change_gbp":         round(cf_bill - base_bill, 4),
                    "pct_change":              round((cf_bill - base_bill) / base_bill * 100, 2)
                                               if base_bill > 0 else 0,
                    "tariff": tariff.plan_name if tariff else "standard",
                    "unit_rate_gbp_per_kwh": rate,
                })

            return json.dumps(out)
        except Exception as e:
            return json.dumps({"error": f"counterfactual_fwd failed: {e}"})

    # ===========================================================
    # 7 — counterfactual_inv
    # ===========================================================
    @tool
    def counterfactual_inv(
        target: float,
        level: str = "consumption",
        constraints_json: str = "{}",
    ) -> str:
        """Inverse what-if: find feature changes needed to reach a target.

        Args:
            target:           Target value to reach.
                              consumption level: watts. bill level: £/month.
                              appliance level: watts for specific appliance.
            level:            "consumption", "bill", or "appliance".
            constraints_json: JSON dict of {feature: [min, max]} bounds.
        """
        try:
            tariff  = _get_tariff()
            rate    = _tariff_rate(tariff)
            profile = session.get("profile")

            # Counterfactual requires multivariate features — auto-initialize if none loaded
            if not feat_cols:
                try:
                    df_cf = pd.read_csv(dataset_path)
                    auto_feats = [c for c in df_cf.select_dtypes(include="number").columns
                                  if c not in (target_column, time_column)][:10]
                    if auto_feats:
                        from tinyts.agent_tools import create_agent_tools
                        _, new_session = create_agent_tools(
                            dataset_path, time_column, target_column,
                            session.get("output_dir", "."), auto_feats,
                        )
                        session.update(new_session)
                        # Re-profile so internal tools see the features
                        _call("profile_dataset", {})
                except Exception:
                    pass

            if level == "bill":
                # Convert £ target saving to watts reduction
                freq_str  = getattr(profile, "inferred_frequency", None) if profile else None
                step_sec  = _freq_to_seconds(freq_str)
                kwh_saving = target / rate
                watt_reduction = kwh_saving * 1000 * 3600 / step_sec if step_sec > 0 else 0
                final_preds = session.get("final_predictions", [])
                current_mean = float(np.mean(final_preds)) if final_preds else (
                    getattr(profile, "mean", 500.0) or 500.0)
                watts_target = max(0, current_mean - watt_reduction)
                r = _call("counterfactual_inverse",
                          {"target_value": watts_target, "constraints_json": constraints_json})
            else:
                r = _call("counterfactual_inverse",
                          {"target_value": target, "constraints_json": constraints_json})

            if "error" in r:
                return json.dumps(r)

            out: Dict[str, Any] = {
                "status": "ok", "level": level,
                "target": target,
                "current_predicted":   r.get("current_predicted"),
                "optimized_predicted": r.get("optimized_predicted"),
                "gap":       r.get("gap"),
                "converged": r.get("converged"),
                "feature_recommendations": r.get("feature_recommendations", {}),
                "model_used": r.get("model"),
            }
            if level == "bill":
                out["bill_target_gbp"] = target
                out["watts_target"]    = round(watts_target, 1)
                out["tariff"]          = tariff.plan_name if tariff else "standard"
                out["unit_rate"]       = rate
            return json.dumps(out)
        except Exception as e:
            return json.dumps({"error": f"counterfactual_inv failed: {e}"})

    # ===========================================================
    # 8 — get_cost_analysis
    # ===========================================================
    @tool
    def get_cost_analysis() -> str:
        """Historical electricity cost breakdown using the active tariff.

        Returns daily/weekly/monthly cost estimates, standing charge breakdown,
        tariff details, and per-appliance cost breakdown if available.
        """
        try:
            tariff  = _get_tariff()
            rate    = _tariff_rate(tariff)
            profile = session.get("profile")

            # Ensure profiled
            if profile is None:
                _call("profile_dataset", {})
                profile = session.get("profile")

            df_csv = pd.read_csv(dataset_path, nrows=100000)
            df_csv[time_column] = pd.to_datetime(df_csv[time_column], errors="coerce")
            df_csv = df_csv.sort_values(time_column).dropna(subset=[time_column])

            target = pd.to_numeric(df_csv[target_column], errors="coerce").dropna()
            freq_str = getattr(profile, "inferred_frequency", None) if profile else None
            step_sec = _freq_to_seconds(freq_str)
            step_h   = step_sec / 3600

            total_kwh    = float(target.sum()) * step_h / 1000
            n_steps      = len(target)
            total_hours  = n_steps * step_h
            daily_kwh    = total_kwh / (total_hours / 24) if total_hours > 0 else 0
            daily_cost   = daily_kwh * rate
            weekly_cost  = daily_cost * 7
            monthly_cost = daily_cost * 30.44

            standing_daily  = tariff.standing_charge_gbp_per_day if tariff else _STANDING
            monthly_standing = standing_daily * 30.44
            monthly_total    = monthly_cost + monthly_standing

            # Per-appliance cost breakdown
            appl_breakdown = []
            for col in [c for c in df_csv.columns if c.startswith("appl_")]:
                vals = pd.to_numeric(df_csv[col], errors="coerce").dropna()
                if vals.empty:
                    continue
                a_kwh = float(vals.sum()) * step_h / 1000
                a_hours = len(vals) * step_h
                a_daily_kwh = a_kwh / (a_hours / 24) if a_hours > 0 else 0
                a_monthly_cost = a_daily_kwh * 30.44 * rate
                appl_breakdown.append({
                    "appliance": col.replace("appl_", "").replace("_", " ").title(),
                    "monthly_kwh": round(a_daily_kwh * 30.44, 2),
                    "monthly_cost_gbp": round(a_monthly_cost, 4),
                    "pct_of_total": round(a_monthly_cost / monthly_cost * 100, 1)
                                    if monthly_cost > 0 else 0,
                })
            appl_breakdown.sort(key=lambda x: -x["monthly_cost_gbp"])

            return json.dumps({
                "status": "ok",
                "tariff": tariff.plan_name if tariff else "standard",
                "supplier": getattr(tariff, "supplier", "unknown") if tariff else "unknown",
                "unit_rate_gbp_per_kwh": rate,
                "standing_charge_gbp_per_day": standing_daily,
                "total_kwh_in_dataset": round(total_kwh, 2),
                "daily_kwh_avg": round(daily_kwh, 3),
                "daily_cost_gbp": round(daily_cost, 4),
                "weekly_cost_gbp": round(weekly_cost, 4),
                "monthly_energy_cost_gbp": round(monthly_cost, 4),
                "monthly_standing_charge_gbp": round(monthly_standing, 4),
                "monthly_total_gbp": round(monthly_total, 4),
                "annual_estimate_gbp": round(monthly_total * 12, 2),
                "appliance_cost_breakdown": appl_breakdown,
            })
        except Exception as e:
            return json.dumps({"error": f"get_cost_analysis failed: {e}"})

    # ===========================================================
    # 9 — get_consumption_analysis
    # ===========================================================
    @tool
    def get_consumption_analysis(level: str = "home") -> str:
        """Statistical consumption analysis with patterns and natural-language explanation.

        Args:
            level: "home"      — whole-home electricity statistics + hourly/daily patterns
                   "room"      — per-room temperature/humidity breakdown
                   "appliance" — per-appliance usage stats, ranked by consumption
                   "all"       — all three levels combined
        """
        try:
            df_csv = pd.read_csv(dataset_path, nrows=100000)
            df_csv[time_column] = pd.to_datetime(df_csv[time_column], errors="coerce")
            df_csv = df_csv.sort_values(time_column).dropna(subset=[time_column])

            profile = session.get("profile")
            freq_str = getattr(profile, "inferred_frequency", None) if profile else None
            step_h = _freq_to_seconds(freq_str) / 3600

            result: Dict[str, Any] = {"status": "ok", "level": level}

            # ── Home level ──────────────────────────────────────────────
            if level in ("home", "all"):
                elec = pd.to_numeric(df_csv[target_column], errors="coerce").dropna()
                n_days = len(elec) * step_h / 24

                # Hourly profile
                df_csv["_hour"] = df_csv[time_column].dt.hour
                hourly_mean = df_csv.groupby("_hour")[target_column].apply(
                    lambda s: pd.to_numeric(s, errors="coerce").mean()
                ).to_dict()
                peak_hour = int(max(hourly_mean, key=hourly_mean.get)) if hourly_mean else 0

                # Day-of-week
                df_csv["_dow"] = df_csv[time_column].dt.dayofweek
                dow_mean = df_csv.groupby("_dow")[target_column].apply(
                    lambda s: pd.to_numeric(s, errors="coerce").mean()
                ).to_dict()
                peak_dow = int(max(dow_mean, key=dow_mean.get)) if dow_mean else 0
                dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

                # Weekly pattern (weekend vs weekday)
                weekday_w = float(np.mean([v for d, v in dow_mean.items() if d < 5]))
                weekend_w = float(np.mean([v for d, v in dow_mean.items() if d >= 5]))

                result["home"] = {
                    "mean_watts": round(float(elec.mean()), 1),
                    "std_watts":  round(float(elec.std()), 1),
                    "min_watts":  round(float(elec.min()), 1),
                    "max_watts":  round(float(elec.max()), 1),
                    "p95_watts":  round(float(elec.quantile(0.95)), 1),
                    "monthly_kwh_estimate": round(float(elec.mean()) / 1000 * 24 * 30.44, 2),
                    "n_days_of_data": round(n_days, 1),
                    "peak_hour": peak_hour,
                    "hourly_profile_watts": {str(h): round(float(v), 1)
                                             for h, v in hourly_mean.items()},
                    "peak_day_of_week": dow_names[peak_dow] if peak_dow < 7 else "?",
                    "weekday_avg_watts": round(weekday_w, 1),
                    "weekend_avg_watts": round(weekend_w, 1),
                    "weekend_vs_weekday_pct": round((weekend_w - weekday_w) / weekday_w * 100, 1)
                                              if weekday_w > 0 else 0,
                    "narrative": (
                        f"Peak consumption at {peak_hour}:00 "
                        f"({'evening' if 17 <= peak_hour <= 22 else 'daytime' if 8 <= peak_hour <= 16 else 'night'} peak). "
                        f"Weekend use is {abs(round((weekend_w-weekday_w)/weekday_w*100,1))}% "
                        f"{'higher' if weekend_w > weekday_w else 'lower'} than weekdays."
                    ) if hourly_mean else "Insufficient data.",
                }

            # ── Room level ──────────────────────────────────────────────
            if level in ("room", "all"):
                temp_cols = [c for c in df_csv.columns if c.startswith("temp_room_")]
                hum_col   = "humidity_mean" if "humidity_mean" in df_csv.columns else None
                TEMP_LOW, TEMP_HIGH = 180, 220  # IDEAL stores °C×10
                rooms = []
                for col in temp_cols:
                    vals = pd.to_numeric(df_csv[col], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    in_band = float(((vals >= TEMP_LOW) & (vals <= TEMP_HIGH)).mean() * 100)
                    rooms.append({
                        "sensor": col,
                        "mean_temp_c": round(float(vals.mean()) / 10, 1),
                        "in_comfort_band_pct": round(in_band, 1),
                        "below_18c_pct": round(float((vals < TEMP_LOW).mean() * 100), 1),
                        "above_22c_pct": round(float((vals > TEMP_HIGH).mean() * 100), 1),
                    })

                hum_result = None
                if hum_col:
                    hvals = pd.to_numeric(df_csv[hum_col], errors="coerce").dropna()
                    if not hvals.empty:
                        hum_result = {
                            "mean_pct": round(float(hvals.mean()), 1),
                            "in_band_pct": round(float(((hvals >= 40) & (hvals <= 60)).mean() * 100), 1),
                        }

                result["room"] = {
                    "rooms": rooms,
                    "humidity": hum_result,
                    "comfort_thresholds": "18–22°C, 40–60% RH",
                    "note": "IDEAL temperatures stored as °C×10 internally",
                }

            # ── Appliance level ─────────────────────────────────────────
            if level in ("appliance", "all"):
                tariff = _get_tariff()
                rate   = _tariff_rate(tariff)
                appl_cols = [c for c in df_csv.columns if c.startswith("appl_")]
                appliances = []
                home_mean = float(
                    pd.to_numeric(df_csv[target_column], errors="coerce").mean()
                )
                for col in appl_cols:
                    vals = pd.to_numeric(df_csv[col], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    mean_w = float(vals.mean())
                    monthly_kwh  = mean_w / 1000 * 24 * 30.44
                    monthly_cost = monthly_kwh * rate
                    on_pct = float((vals > 10).mean() * 100)
                    appliances.append({
                        "appliance": col.replace("appl_", "").replace("_", " ").title(),
                        "mean_watts": round(mean_w, 1),
                        "monthly_kwh": round(monthly_kwh, 2),
                        "monthly_cost_gbp": round(monthly_cost, 4),
                        "pct_of_home": round(mean_w / home_mean * 100, 1) if home_mean > 0 else 0,
                        "on_fraction_pct": round(on_pct, 1),
                    })
                appliances.sort(key=lambda x: -x["monthly_cost_gbp"])
                result["appliance"] = {
                    "appliances": appliances,
                    "total_tracked_monthly_gbp": round(
                        sum(a["monthly_cost_gbp"] for a in appliances), 4),
                    "tariff": tariff.plan_name if tariff else "standard",
                }

            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": f"get_consumption_analysis failed: {e}"})

    # ===========================================================
    # 10 — make_budget
    # ===========================================================
    @tool
    def make_budget(budget_gbp: Optional[float] = None) -> str:
        """Two-phase budget tool.

        Phase 1 — budget_gbp=None (default):
            Shows historical consumption patterns, appliance breakdown,
            yearly projection. Sets session["awaiting_budget_input"]=True.
            Saves an appliance pie chart as budget_pie.html.

        Phase 2 — budget_gbp=<number>:
            Saves the budget to session, computes gap, returns ranked
            appliance-level suggestions to stay within budget.
        """
        try:
            tariff = _get_tariff()
            rate   = _tariff_rate(tariff)

            profile = session.get("profile")
            if profile is None:
                _call("profile_dataset", {})
                profile = session.get("profile")

            mean_w   = getattr(profile, "mean", 500.0) or 500.0
            freq_str = getattr(profile, "inferred_frequency", None) if profile else None
            step_sec = _freq_to_seconds(freq_str)
            step_h   = step_sec / 3600
            standing_daily = getattr(tariff, "standing_charge_gbp_per_day", _STANDING)

            monthly_kwh    = mean_w / 1000 * 24 * 30.44
            monthly_energy = monthly_kwh * rate
            monthly_total  = monthly_energy + standing_daily * 30.44
            yearly_total   = monthly_total * 12

            # Appliance breakdown from CSV
            appl_data: List[Dict[str, Any]] = []
            try:
                df_csv = pd.read_csv(dataset_path, nrows=50000)
                for col in [c for c in df_csv.columns if c.startswith("appl_")]:
                    vals = pd.to_numeric(df_csv[col], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    a_mean_w = float(vals.mean())
                    a_monthly_kwh = a_mean_w / 1000 * 24 * 30.44
                    a_cost = a_monthly_kwh * rate
                    appl_data.append({
                        "appliance": col.replace("appl_", "").replace("_", " ").title(),
                        "appliance_key": col.replace("appl_", ""),
                        "monthly_kwh": round(a_monthly_kwh, 2),
                        "monthly_cost_gbp": round(a_cost, 4),
                        "pct_of_home": round(a_mean_w / mean_w * 100, 1) if mean_w > 0 else 0,
                    })
                appl_data.sort(key=lambda x: -x["monthly_cost_gbp"])
            except Exception:
                pass

            # Save pie chart
            out_dir = session.get("output_dir", ".")
            try:
                import plotly.graph_objects as go
                if appl_data:
                    labels = [a["appliance"] for a in appl_data]
                    values = [a["monthly_cost_gbp"] for a in appl_data]
                    fig = go.Figure(go.Pie(
                        labels=labels, values=values,
                        hole=0.4,
                        textinfo="label+percent",
                        hovertemplate="%{label}<br>£%{value:.4f}/month<extra></extra>",
                    ))
                    fig.update_layout(
                        title="Monthly Electricity Cost by Appliance",
                        template="plotly_white",
                    )
                    fig.write_html(str(Path(out_dir) / "budget_pie.html"))
            except Exception:
                pass

            # ── Phase 1: show analysis, ask for budget ──
            if budget_gbp is None:
                session["awaiting_budget_input"] = True
                return json.dumps({
                    "status": "ok",
                    "phase": 1,
                    "awaiting_budget_input": True,
                    "current_monthly_total_gbp": round(monthly_total, 4),
                    "current_yearly_estimate_gbp": round(yearly_total, 2),
                    "monthly_energy_cost_gbp": round(monthly_energy, 4),
                    "monthly_standing_charge_gbp": round(standing_daily * 30.44, 4),
                    "monthly_kwh": round(monthly_kwh, 2),
                    "tariff": tariff.plan_name if tariff else "standard",
                    "appliance_breakdown": appl_data,
                    "message": (
                        "Budget analysis complete. Pie chart saved. "
                        "Please enter your monthly budget target in the input box below."
                    ),
                })

            # ── Phase 2: budget set → suggestions ──
            session["awaiting_budget_input"] = False
            session["monthly_budget_gbp"] = float(budget_gbp)

            overshoot = max(0, monthly_total - budget_gbp)
            suggestions = []

            if overshoot > 0 and appl_data:
                for appl in appl_data[:5]:
                    name = appl["appliance"]
                    cost = appl["monthly_cost_gbp"]
                    schedulable = any(k in name.lower()
                                      for k in ("washing", "dishwasher", "ev", "charger"))
                    suggestions.append({
                        "action": (f"Shift {name} to off-peak hours (00:30–07:30)"
                                   if schedulable else f"Reduce {name} usage by 15%"),
                        "appliance": name,
                        "current_cost_gbp": cost,
                        "potential_saving_gbp": round(cost * (0.30 if schedulable else 0.15), 4),
                        "effort": "low" if schedulable else "medium",
                    })
            elif overshoot > 0:
                suggestions = [
                    {"action": "Shift flexible loads to off-peak",
                     "potential_saving_gbp": round(overshoot * 0.30, 4), "effort": "low"},
                    {"action": "Reduce heating setpoint by 1°C",
                     "potential_saving_gbp": round(overshoot * 0.20, 4), "effort": "low"},
                ]

            return json.dumps({
                "status": "ok",
                "phase": 2,
                "monthly_budget_gbp": budget_gbp,
                "projected_monthly_gbp": round(monthly_total, 4),
                "overshoot_gbp": round(overshoot, 4),
                "on_track": overshoot <= 0,
                "suggestions": suggestions,
                "tariff": tariff.plan_name if tariff else "standard",
            })
        except Exception as e:
            return json.dumps({"error": f"make_budget failed: {e}"})

    # ===========================================================
    # 11 — weather_impact_analysis
    # ===========================================================
    @tool
    def weather_impact_analysis() -> str:
        """Longitudinal analysis of how weather/temperature drift has been impacting
        energy consumption at home, room, and appliance level.

        Computes Pearson correlations, linear regression slopes (watts per °C),
        and per-temperature-band consumption tables for each available weather feature.
        """
        try:
            df_csv = pd.read_csv(dataset_path)
            df_csv[time_column] = pd.to_datetime(df_csv[time_column], errors="coerce")

            elec = pd.to_numeric(df_csv.get(target_column, pd.Series(dtype=float)), errors="coerce")

            weather_map = {
                "temperature_mean":   "Indoor temperature (°C×10)",
                "temp_room_1":        "Room 1 temperature (°C×10)",
                "humidity_mean":      "Indoor humidity (%)",
                "gas_watts":          "Gas consumption (W proxy)",
                "weather_temp_c":     "Outdoor temperature (°C)",
                "weather_humidity_pct": "Outdoor humidity (%)",
            }

            drivers = []
            for col, label in weather_map.items():
                if col not in df_csv.columns:
                    continue
                feat  = pd.to_numeric(df_csv[col], errors="coerce")
                valid = elec.notna() & feat.notna()
                if valid.sum() < 30:
                    continue
                e_v, f_v = elec[valid].values, feat[valid].values
                corr  = float(np.corrcoef(f_v, e_v)[0, 1])
                slope = float(np.polyfit(f_v, e_v, 1)[0])

                bands = pd.cut(feat[valid], bins=5)
                band_table = elec[valid].groupby(bands).mean().dropna()
                bands_out  = [{"band": str(b), "mean_watts": round(float(v), 1)}
                              for b, v in band_table.items()]

                # Appliance-level correlation
                appl_corrs = {}
                for ac in [c for c in df_csv.columns if c.startswith("appl_")]:
                    av = pd.to_numeric(df_csv[ac], errors="coerce")
                    av_valid = feat.notna() & av.notna()
                    if av_valid.sum() < 30:
                        continue
                    ac_corr = float(np.corrcoef(feat[av_valid].values, av[av_valid].values)[0, 1])
                    if abs(ac_corr) > 0.15:
                        appl_corrs[ac.replace("appl_", "")] = round(ac_corr, 3)

                drivers.append({
                    "feature":    col,
                    "label":      label,
                    "pearson_corr_with_electricity": round(corr, 3),
                    "regression_slope_watts_per_unit": round(slope, 3),
                    "n_samples":  int(valid.sum()),
                    "consumption_by_band": bands_out,
                    "appliance_correlations": appl_corrs,
                    "interpretation": (
                        f"{'Positive' if corr > 0 else 'Negative'} correlation ({corr:+.3f}): "
                        f"each unit increase in {label} → {slope:+.1f} W on average. "
                        + ("Heating-dominated system." if col == "temperature_mean" and corr < -0.3
                           else "")
                    ),
                })

            if not drivers:
                return json.dumps({"error": "No weather/temperature features found. "
                                            "Ensure temperature_mean or weather_temp_c are exported."})

            drivers.sort(key=lambda x: -abs(x["pearson_corr_with_electricity"]))
            return json.dumps({
                "status": "ok",
                "weather_drivers": drivers,
                "top_driver": drivers[0]["feature"],
                "note": "Indoor IDEAL temperatures stored as °C×10; outdoor weather in actual °C.",
            })
        except Exception as e:
            return json.dumps({"error": f"weather_impact_analysis failed: {e}"})

    # ===========================================================
    # 12 — suggest_budget_correction
    # ===========================================================
    @tool
    def suggest_budget_correction() -> str:
        """Appliance-level breakdown of how to stay within the budget set by make_budget.

        Requires make_budget to have been called with a budget_gbp value first.
        Returns ranked corrective actions with real measured appliance costs.
        """
        try:
            budget = session.get("monthly_budget_gbp")
            if budget is None:
                return json.dumps({
                    "error": "No budget set. Call make_budget(budget_gbp=<amount>) first.",
                    "hint": "Example: make_budget(budget_gbp=100)",
                })

            tariff = _get_tariff()
            rate   = _tariff_rate(tariff)
            profile = session.get("profile")
            mean_w = getattr(profile, "mean", 500.0) or 500.0
            freq_str = getattr(profile, "inferred_frequency", None) if profile else None
            step_h   = _freq_to_seconds(freq_str) / 3600
            standing_daily = getattr(tariff, "standing_charge_gbp_per_day", _STANDING)
            monthly_total  = mean_w / 1000 * 24 * 30.44 * rate + standing_daily * 30.44
            overshoot = max(0, monthly_total - budget)

            appl_data: List[Dict[str, Any]] = []
            try:
                df_csv = pd.read_csv(dataset_path, nrows=50000)
                for col in [c for c in df_csv.columns if c.startswith("appl_")]:
                    vals = pd.to_numeric(df_csv[col], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    a_mean_w = float(vals.mean())
                    a_cost   = a_mean_w / 1000 * 24 * 30.44 * rate
                    on_pct   = float((vals > 10).mean() * 100)

                    # Detect if shift-able: washing/dishwasher/EV = schedulable
                    name = col.replace("appl_", "").lower()
                    schedulable = any(k in name for k in ("washing", "dishwasher", "ev", "charger"))
                    standby     = any(k in name for k in ("fridge", "freezer", "router"))

                    if schedulable:
                        action = f"Shift {name.title()} to off-peak (save ~30%)"
                        saving = a_cost * 0.30
                    elif standby:
                        action = f"Upgrade {name.title()} or reduce setpoint (save ~15%)"
                        saving = a_cost * 0.15
                    else:
                        action = f"Reduce {name.title()} usage by 20%"
                        saving = a_cost * 0.20

                    appl_data.append({
                        "appliance": name.title(),
                        "monthly_cost_gbp": round(a_cost, 4),
                        "pct_of_home": round(a_mean_w / mean_w * 100, 1) if mean_w > 0 else 0,
                        "on_fraction_pct": round(on_pct, 1),
                        "action": action,
                        "potential_saving_gbp": round(saving, 4),
                        "effort": "low" if schedulable else "medium",
                    })
                appl_data.sort(key=lambda x: -x["monthly_cost_gbp"])
            except Exception:
                pass

            total_achievable = sum(a["potential_saving_gbp"] for a in appl_data)
            return json.dumps({
                "status": "ok",
                "monthly_budget_gbp":    budget,
                "projected_monthly_gbp": round(monthly_total, 4),
                "overshoot_gbp":         round(overshoot, 4),
                "on_track":              overshoot <= 0,
                "total_achievable_saving_gbp": round(total_achievable, 4),
                "appliance_actions": appl_data,
                "tariff": tariff.plan_name if tariff else "standard",
                "data_source": "real_appliance_data" if appl_data else "profile_estimate",
            })
        except Exception as e:
            return json.dumps({"error": f"suggest_budget_correction failed: {e}"})

    # ===========================================================
    # 13 — evaluate_tariff_switch
    # ===========================================================
    @tool
    def evaluate_tariff_switch(alternative_tariff_id: str = "economy7") -> str:
        """Replay historical consumption against an alternative tariff.

        Args:
            alternative_tariff_id: Tariff ID to compare against (default "economy7").
                Available options depend on configured tariff cache.
        Returns projected annual saving/cost and recommendation.
        """
        try:
            from energyx.data.tariffs import get_tariff_cache
            cache   = get_tariff_cache()
            current = cache.get_active_tariff()
            alt     = next((t for t in cache.list_tariffs()
                            if t.tariff_id == alternative_tariff_id), None)

            if alt is None:
                available = [t.tariff_id for t in cache.list_tariffs()]
                return json.dumps({"error": f"Tariff '{alternative_tariff_id}' not found.",
                                   "available_tariffs": available})

            profile = session.get("profile")
            if profile is None:
                _call("profile_dataset", {})
                profile = session.get("profile")

            mean_w    = getattr(profile, "mean", 500.0) or 500.0
            annual_kwh = mean_w / 1000.0 * 8760

            curr_rate    = _tariff_rate(current)
            alt_rate     = _tariff_rate(alt)
            curr_standing = getattr(current, "standing_charge_gbp_per_day", _STANDING) * 365
            alt_standing  = getattr(alt,     "standing_charge_gbp_per_day", _STANDING) * 365

            curr_annual = annual_kwh * curr_rate + curr_standing
            alt_annual  = annual_kwh * alt_rate  + alt_standing
            saving      = curr_annual - alt_annual

            return json.dumps({
                "status": "ok",
                "current_tariff":  current.plan_name if current else "unknown",
                "alternative_tariff": alt.plan_name,
                "annual_kwh_estimate": round(annual_kwh, 1),
                "current_annual_cost_gbp":     round(curr_annual, 2),
                "alternative_annual_cost_gbp": round(alt_annual, 2),
                "projected_annual_saving_gbp": round(saving, 2),
                "recommendation": "switch" if saving > 0 else "stay",
                "note": "Based on profile mean consumption.",
            })
        except Exception as e:
            return json.dumps({"error": f"evaluate_tariff_switch failed: {e}"})

    # ===========================================================
    # 14 — generate_report
    # ===========================================================
    @tool
    def generate_report() -> str:
        """Generate a comprehensive markdown report of all analyses performed in this session.

        Writes to {output_dir}/report_{timestamp}.md.
        Call LAST — after forecasting, anomaly detection, or any analysis.
        """
        try:
            from tinyts.config import get_llm

            parts: List[str] = []
            profile = session.get("profile")
            if profile:
                parts.append(
                    f"DATASET: {profile.shape[0]} rows, freq={profile.inferred_frequency}, "
                    f"mean={round(profile.mean, 2)}W, seasonality={profile.seasonal_period}, "
                    f"trend={profile.has_trend}"
                )

            model_results = session.get("model_results", {})
            if model_results:
                lines = [f"  {n}: MAPE={d['mean_mape']:.3f}% (±{d['std_mape']:.3f}%)"
                         for n, d in sorted(model_results.items(), key=lambda x: x[1]["mean_mape"])]
                parts.append("MODEL RESULTS:\n" + "\n".join(lines))

            strat = session.get("ensemble_strategy")
            if strat:
                parts.append(f"FORECAST STRATEGY: {strat['strategy_type']} "
                             f"— best model(s): {strat['selected_models']}")

            anom = session.get("anomaly_results")
            if anom:
                parts.append(f"ANOMALY DETECTION: {anom['n_anomalies']} anomalies, "
                             f"method counts: {anom['method_counts']}")

            for name, exp in session.get("model_explanations", {}).items():
                fi = exp.get("feature_importance_pct", {})
                if fi:
                    top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
                    parts.append(f"FEATURE IMPORTANCE ({name}): "
                                 + ", ".join(f"{k}={v}%" for k, v in top5))

            budget = session.get("monthly_budget_gbp")
            if budget:
                parts.append(f"BUDGET: £{budget:.2f}/month target set.")

            shared = session.get("_shared_explain", {})
            if shared.get("lag_hints"):
                parts.append("LAG ANALYSIS: " + "; ".join(shared["lag_hints"]))

            ctx = "\n\n".join(parts)
            has_results = model_results or anom or shared
            if not has_results:
                return json.dumps({"error": "No analysis results to report yet."})

            prompt = (
                "You are an energy data analyst. Write a concise report (400–600 words) "
                "in markdown format for a home energy analysis session.\n\n"
                f"Session data:\n{ctx}\n\n"
                "Include sections: ## Summary, ## Key Findings, ## Forecast Results (if any), "
                "## Anomalies (if any), ## Recommendations. "
                "Be specific — cite actual numbers. Write in plain English for a homeowner."
            )

            try:
                llm = get_llm(temperature=0.3)
                report_text = llm.invoke(prompt).content
            except Exception as e:
                report_text = f"Report generation failed: {e}\n\n---\n\n{ctx}"

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_dir = session.get("output_dir", ".")
            report_path = Path(out_dir) / f"report_{ts}.md"
            report_path.write_text(f"# EnergyX Analysis Report\n\n{report_text}")
            session["report"] = report_text

            return json.dumps({
                "status": "ok",
                "report_path": str(report_path),
                "report_preview": report_text[:500] + "…" if len(report_text) > 500 else report_text,
            })
        except Exception as e:
            return json.dumps({"error": f"generate_report failed: {e}"})

    # ------------------------------------------------------------------
    return [
        forecast_univariate,
        forecast_multivariate,
        explain_forecast,
        detect_anomalies,
        explain_anomalies,
        counterfactual_fwd,
        counterfactual_inv,
        get_cost_analysis,
        get_consumption_analysis,
        make_budget,
        weather_impact_analysis,
        suggest_budget_correction,
        evaluate_tariff_switch,
        generate_report,
    ]
