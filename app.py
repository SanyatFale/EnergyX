"""
TinyTS-Scientist - Streamlit Application
=========================================
Interactive chat UI using central LLM agent with tool calling.
No LangGraph — the agent calls tools directly.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from tinyts.config import settings

# =========================================================================
# Page config
# =========================================================================
st.set_page_config(
    page_title="TinyTS-Scientist",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { padding: 1rem; }
    .stChatMessage { padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; }
    h1 { color: #1E3A5F; font-weight: 700; }
    .stButton > button { border-radius: 20px; padding: 0.5rem 2rem; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================================
# Session state defaults
# =========================================================================
_DEFAULTS: Dict = {
    "messages": [
        {
            "role": "assistant",
            "content": (
                "**Welcome to TinyTS-Scientist!**\n\n"
                "Upload a CSV in the sidebar or pick one from the data/ folder, "
                "then tell me what you'd like to do:\n\n"
                "- *Forecast energy for next 7 days*\n"
                "- *Detect anomalies in temperature*\n"
                "- *Explain the forecast*\n"
            ),
        }
    ],
    "stage": "dataset_select",
    "show_reasoning": False,
    "agent": None,
    "task_plan": None,
    "dataset_path": None,
    "time_column": None,
    "target_column": None,
    "feature_columns": [],
    "plan_approved": False,
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# =========================================================================
# Helpers
# =========================================================================
def _render_plot(html_str: str, key: str):
    if html_str:
        components.html(html_str, height=500, scrolling=True)


def _read_plot(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _list_data_files() -> List[str]:
    data_dir = settings.data_dir
    files = []
    for ext in ("*.csv", "*.parquet", "*.xlsx"):
        files.extend(data_dir.rglob(ext))
    return sorted(str(f) for f in files)


# =========================================================================
# Sidebar
# =========================================================================
with st.sidebar:
    st.title("TinyTS-Scientist")
    st.caption("Agentic Time-Series Analysis")
    st.divider()

    # --- Dataset selection ---
    st.subheader("Dataset")
    upload = st.file_uploader("Upload CSV / Parquet", type=["csv", "parquet", "xlsx"])
    existing_files = _list_data_files()
    selected_file = st.selectbox(
        "Or pick from data/",
        options=[""] + existing_files,
        index=0,
        format_func=lambda x: Path(x).name if x else "-- select --",
    )

    if upload is not None:
        save_path = settings.data_dir / "raw" / upload.name
        save_path.write_bytes(upload.getvalue())
        st.session_state.dataset_path = str(save_path)
        st.success(f"Uploaded {upload.name}")
    elif selected_file:
        st.session_state.dataset_path = selected_file

    # --- Column config ---
    if st.session_state.dataset_path:
        try:
            df_peek = pd.read_csv(st.session_state.dataset_path, nrows=5)
        except Exception:
            try:
                df_peek = pd.read_parquet(st.session_state.dataset_path).head(5)
            except Exception:
                df_peek = None

        if df_peek is not None:
            cols = list(df_peek.columns)
            st.session_state.time_column = st.selectbox("Time column", options=cols, index=0)
            target_idx = min(1, len(cols) - 1)
            st.session_state.target_column = st.selectbox(
                "Target column", options=cols, index=target_idx
            )
            other_cols = [
                c for c in cols
                if c not in (st.session_state.time_column, st.session_state.target_column)
            ]
            st.session_state.feature_columns = st.multiselect(
                "Feature columns (optional)", options=other_cols
            )

    st.divider()

    # --- Stage indicator ---
    st.subheader("Pipeline Stage")
    st.info(st.session_state.stage.replace("_", " ").title())

    # --- Settings ---
    st.subheader("Settings")
    st.session_state.show_reasoning = st.checkbox(
        "Show reasoning", value=st.session_state.show_reasoning
    )

    st.divider()
    if st.button("Reset conversation", width="stretch"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# =========================================================================
# Main area
# =========================================================================
st.title("TinyTS-Scientist")
st.caption("Agentic Time-Series Forecasting & Anomaly Detection")

# --- Chat history ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("plot_html"):
            _render_plot(msg["plot_html"], f"plot_{i}")
        if st.session_state.show_reasoning and msg.get("reasoning"):
            with st.expander("Reasoning"):
                st.code(msg["reasoning"], language=None)
        if msg.get("dataframe") is not None:
            st.dataframe(msg["dataframe"], width="stretch")

# =========================================================================
# Human approval gate (editable plan form)
# =========================================================================
_UNIVARIATE_ONLY = ["Naive", "SeasonalNaive", "ARIMA", "ETS", "N-BEATS", "TinyTimeMixer"]
_MULTIVARIATE_ONLY = []  # No models are multivariate-only
_BOTH = ["RandomForest", "LightGBM"]  # Support both uni and multivariate
_ALL_MODELS = _UNIVARIATE_ONLY + _BOTH

if st.session_state.stage == "awaiting_approval" and not st.session_state.plan_approved:
    plan = st.session_state.task_plan
    if plan:
        st.subheader("Edit & Approve Plan")

        if plan.reasoning:
            st.caption(f"LLM reasoning: {plan.reasoning}")

        # Multivariate checkbox outside form to allow reactive updates
        is_multivariate = st.checkbox(
            "Multivariate", value=plan.is_multivariate, key="mv_checkbox"
        )

        with st.form("plan_form"):
            col1, col2 = st.columns(2)
            with col1:
                task_type = st.selectbox(
                    "Task type",
                    options=["forecast", "anomaly", "both"],
                    index=["forecast", "anomaly", "both"].index(plan.task_type),
                )
                horizon = st.number_input(
                    "Horizon (time steps)",
                    min_value=1, max_value=1000,
                    value=plan.horizon or 12,
                    disabled=(plan.task_type == "anomaly"),
                )
            with col2:
                if plan.task_type != "anomaly":
                    # Filter available models based on multivariate flag
                    if is_multivariate:
                        available_models = _BOTH
                        default_models = [m for m in (plan.models_included or _BOTH) if m in _BOTH]
                    else:
                        available_models = _ALL_MODELS
                        default_models = [m for m in (plan.models_included or _UNIVARIATE_ONLY[:3]) if m in _ALL_MODELS]

                    models_included = st.multiselect(
                        "Forecast models", options=available_models, default=default_models,
                    )
                else:
                    st.info("Anomaly detection uses 7-method ensemble")
                    models_included = []
                needs_cv = st.checkbox("Cross-validation", value=plan.needs_cv,
                                       disabled=(plan.task_type == "anomaly"))
                needs_explanation = st.checkbox("Explanation", value=plan.needs_explanation)
                needs_report = st.checkbox("Report", value=plan.needs_report)

            submitted = st.form_submit_button("Approve & Run", type="primary")

            if submitted:
                plan.task_type = task_type
                plan.horizon = horizon
                plan.is_multivariate = is_multivariate
                plan.models_included = models_included
                plan.needs_cv = needs_cv
                plan.needs_explanation = needs_explanation
                plan.needs_report = needs_report
                st.session_state.task_plan = plan

                edit_msg = (
                    f"**Plan approved:** {task_type} | "
                    f"horizon={horizon} | "
                    f"models={models_included or 'anomaly ensemble'}"
                )
                st.session_state.messages.append({"role": "assistant", "content": edit_msg})
                st.session_state.plan_approved = True
                st.rerun()

    st.stop()

# =========================================================================
# Pipeline execution after approval — agent tool-calling loop
# =========================================================================
if st.session_state.plan_approved and st.session_state.stage == "awaiting_approval":
    agent = st.session_state.agent
    plan = st.session_state.task_plan

    with st.chat_message("assistant"):
        with st.spinner("Running agent..."):
            try:
                tool_log = []

                def _on_tool(name, args, result_str):
                    """Progress callback from agent loop."""
                    tool_log.append((name, args, result_str))

                result = agent.execute(plan, on_tool_call=_on_tool)
                session = result["session"]
                out_dir = result["output_dir"]

                # --- Show per-tool results ---
                for tname, targs, tres in tool_log:
                    if tname == "train_forecast_model" or tname == "train_and_explain_forecast":
                        try:
                            r = json.loads(tres)
                            msg = (
                                f"**{r.get('model_name', tname)}**: "
                                f"MAPE={r.get('mean_mape', '?')}% "
                                f"(+/-{r.get('std_mape', '?')}%)"
                            )
                            if r.get("warning"):
                                msg += f"\n  {r['warning']}"
                            st.markdown(msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": msg}
                            )
                        except Exception:
                            pass

                    elif tname == "select_ensemble_strategy":
                        try:
                            r = json.loads(tres)
                            msg = (
                                f"**Strategy:** {r.get('strategy_type', '?')} — "
                                f"{', '.join(r.get('selected_models', []))}"
                            )
                            st.markdown(msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": msg}
                            )
                        except Exception:
                            pass

                    elif tname == "detect_anomalies":
                        try:
                            r = json.loads(tres)
                            msg = f"**Anomalies:** {r.get('n_anomalies', 0)} found"
                            st.markdown(msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": msg}
                            )
                        except Exception:
                            pass

                # --- Show plots ---
                forecast_html = _read_plot(str(Path(out_dir) / "final_forecast.html"))
                if forecast_html:
                    _render_plot(forecast_html, f"fc_{time.time()}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Forecast plot:", "plot_html": forecast_html}
                    )

                comparison_html = _read_plot(str(Path(out_dir) / "model_comparison.html"))
                if comparison_html:
                    _render_plot(comparison_html, f"cmp_{time.time()}")

                anomaly_html = _read_plot(str(Path(out_dir) / "anomaly_detection.html"))
                if anomaly_html:
                    _render_plot(anomaly_html, f"anom_{time.time()}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Anomaly plot:", "plot_html": anomaly_html}
                    )

                # --- Show LLM summary ---
                agent_response = result.get("response", "")
                if agent_response:
                    st.markdown(f"**Agent Summary:**\n\n{agent_response}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"**Agent Summary:**\n\n{agent_response}"}
                    )

                # --- Show report ---
                report = session.get("report")
                if report:
                    with st.expander("Full Report"):
                        st.markdown(report)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Report generated. Expand above to view."}
                    )

                # --- Show explainability ---
                for name, exp in session.get("model_explanations", {}).items():
                    fi = exp.get("feature_importance", {})
                    shap_v = exp.get("shap_values", {})
                    if fi or shap_v:
                        with st.expander(f"Explainability: {name}"):
                            if fi:
                                st.markdown("**Feature Importance:**")
                                for k, v in sorted(fi.items(), key=lambda x: -abs(x[1]))[:10]:
                                    st.markdown(f"- {k}: {v:.4f}")
                            if shap_v:
                                st.markdown("**SHAP Values:**")
                                for k, v in sorted(shap_v.items(), key=lambda x: -abs(x[1]))[:10]:
                                    st.markdown(f"- {k}: {v:.4f}")

                st.session_state.stage = "done"
                st.markdown("Pipeline complete.")

            except Exception as e:
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

    st.session_state.plan_approved = False
    st.rerun()

# =========================================================================
# Chat input — Profile + Query Understanding
# =========================================================================
prompt = st.chat_input("What would you like to do?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Ensure dataset is configured
    if (
        not st.session_state.dataset_path
        or not st.session_state.time_column
        or not st.session_state.target_column
    ):
        resp = "Please select a dataset and configure the time/target columns in the sidebar first."
        st.session_state.messages.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"):
            st.markdown(resp)
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                from tinyts.agent import TinyTSAgent

                # Create agent (profiles dataset inside plan())
                agent = TinyTSAgent(
                    dataset_path=st.session_state.dataset_path,
                    time_column=st.session_state.time_column,
                    target_column=st.session_state.target_column,
                    feature_columns=st.session_state.feature_columns or None,
                )
                st.session_state.agent = agent
                st.session_state.stage = "profiling"

                # Plan (also runs profile_dataset internally)
                st.session_state.stage = "query_understanding"
                task_plan = agent.plan(prompt)
                st.session_state.task_plan = task_plan

                # Show profile (populated by plan() via profile_dataset tool)
                profile = agent.session.get("profile")
                if profile:
                    head_df = pd.DataFrame(profile.head_rows) if profile.head_rows else None
                    profile_msg = f"**Dataset profiled:** {profile.shape[0]} rows x {profile.shape[1]} cols"
                    st.markdown(profile_msg)
                    if head_df is not None:
                        st.dataframe(head_df, width="stretch")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": profile_msg, "dataframe": head_df}
                    )

                plan_text = (
                    f"**Understood:** {task_plan.task_type} | "
                    f"horizon={task_plan.horizon} | "
                    f"multivariate={task_plan.is_multivariate} | "
                    f"models={task_plan.models_included or 'default'}"
                )
                st.markdown(plan_text)
                st.session_state.messages.append({"role": "assistant", "content": plan_text})

                approval_msg = "Please review the plan above and click **Approve & Run** to proceed."
                st.markdown(approval_msg)
                st.session_state.messages.append({"role": "assistant", "content": approval_msg})

                st.session_state.stage = "awaiting_approval"
                st.session_state.plan_approved = False

            except Exception as e:
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

    st.rerun()

# =========================================================================
# Footer
# =========================================================================
st.divider()
st.caption("TinyTS-Scientist")
