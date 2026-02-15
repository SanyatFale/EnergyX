"""Data Profiler Node - Enhanced deterministic data analysis.

Performs data-analyst-level profiling: column info, types, missing data,
statistical tests, seasonality detection, and generates a DataProfile
for UI display.
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from tinyts.nodes.base import BaseNode
from tinyts.state import AgentState, DatasetSummary, DataProfile


class DataProfilerNode(BaseNode):
    """Enhanced deterministic data profiling node.

    Performs:
    - Column-level info (name, dtype, missing%, unique count, samples)
    - Head rows for UI preview
    - Column role auto-detection (datetime, numeric, categorical)
    - Frequency inference
    - Missing data analysis
    - Outlier detection (IQR)
    - Stationarity tests (ADF, KPSS)
    - Seasonality detection (ACF, STL)
    - Plotly visualization generation
    """

    def __init__(self):
        super().__init__("DataProfiler")

    def execute(self, state: AgentState) -> AgentState:
        """Profile the dataset and generate enhanced summary."""
        # Load data
        df = self._load_data(state["dataset_path"])

        # Validate columns
        time_col = state["time_column"]
        target_col = state["target_column"]
        covariates = state.get("covariates", []) or []

        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in dataset")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Prepare time series
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        # Extract target series
        y = df[target_col].values

        # Create output directory
        output_dir = Path(state["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build enhanced profile
        profile = self._build_profile(df, time_col, target_col, covariates, y, output_dir)

        # Also build DatasetSummary for backward compatibility
        summary = self._build_summary(profile)

        # Update state
        state["data_profile"] = profile
        state["dataset_summary"] = summary

        return state

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load dataset from CSV or Parquet."""
        path_obj = Path(path)
        if path_obj.suffix == ".csv":
            return pd.read_csv(path)
        elif path_obj.suffix in [".parquet", ".pq"]:
            return pd.read_parquet(path)
        elif path_obj.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

    def _build_profile(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        covariates: list,
        y: np.ndarray,
        output_dir: Path,
    ) -> DataProfile:
        """Build comprehensive DataProfile."""

        n_rows, n_cols = df.shape
        y_clean = y[~np.isnan(y)]

        # Column-level info
        columns = self._build_column_info(df)

        # Head rows
        head_rows = self._build_head_rows(df, n=10)

        # Column role detection
        datetime_cols, numeric_cols, categorical_cols, categorical_values = self._detect_column_roles(df)

        # Missing data
        missing_pct = (df[target_col].isna().sum() / n_rows) * 100

        # Outlier detection (IQR)
        q1, q3 = np.percentile(y_clean, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (y_clean < lower_bound) | (y_clean > upper_bound)
        outlier_pct = (outliers.sum() / len(y_clean)) * 100

        # Frequency inference
        freq = pd.infer_freq(pd.Series(df[time_col]))

        # Date range
        date_range = (str(df[time_col].min()), str(df[time_col].max()))

        # Stationarity tests
        adf_result = adfuller(y_clean, autolag="AIC")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=InterpolationWarning)
            kpss_result = kpss(y_clean, regression="c", nlags="auto")

        # Seasonality detection
        has_trend, has_seasonality, seasonal_period = self._detect_seasonality(y_clean)

        # Generate Plotly visualizations
        plot_paths = self._generate_plots(df, time_col, target_col, y_clean, output_dir)

        return DataProfile(
            columns=columns,
            head_rows=head_rows,
            shape=(n_rows, n_cols),
            datetime_columns=datetime_cols,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            categorical_values=categorical_values,
            time_column=time_col,
            target_column=target_col,
            covariates=covariates,
            inferred_frequency=freq,
            date_range=date_range,
            missing_pct=missing_pct,
            outlier_pct=outlier_pct,
            mean=float(np.mean(y_clean)),
            std=float(np.std(y_clean)),
            min_val=float(np.min(y_clean)),
            max_val=float(np.max(y_clean)),
            adf_statistic=float(adf_result[0]),
            adf_pvalue=float(adf_result[1]),
            kpss_statistic=float(kpss_result[0]),
            kpss_pvalue=float(kpss_result[1]),
            has_trend=has_trend,
            has_seasonality=has_seasonality,
            seasonal_period=seasonal_period,
            plot_paths=plot_paths,
        )

    def _build_column_info(self, df: pd.DataFrame) -> list[dict]:
        """Build per-column metadata for UI display."""
        columns = []
        for col in df.columns:
            info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_pct": round((df[col].isna().sum() / len(df)) * 100, 2),
                "unique_count": int(df[col].nunique()),
                "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
            }
            columns.append(info)
        return columns

    def _build_head_rows(self, df: pd.DataFrame, n: int = 10) -> list[dict]:
        """First n rows as list of dicts for UI preview."""
        return df.head(n).astype(str).to_dict(orient="records")

    def _detect_column_roles(self, df: pd.DataFrame) -> tuple[list, list, list, dict]:
        """Auto-detect column roles: datetime, numeric, categorical.

        Returns (datetime_cols, numeric_cols, categorical_cols, categorical_values).
        """
        datetime_cols = []
        numeric_cols = []
        categorical_cols = []
        categorical_values: dict[str, list[str]] = {}

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            elif df[col].dtype == "object":
                # Try parsing as datetime
                try:
                    pd.to_datetime(df[col].head(5), format="mixed")
                    datetime_cols.append(col)
                except (ValueError, TypeError):
                    categorical_cols.append(col)
                    unique_vals = df[col].dropna().unique()
                    categorical_values[col] = sorted(
                        [str(v) for v in unique_vals[:20]]
                    )
            else:
                categorical_cols.append(col)

        return datetime_cols, numeric_cols, categorical_cols, categorical_values

    def _build_summary(self, profile: DataProfile) -> DatasetSummary:
        """Build DatasetSummary from DataProfile for backward compatibility."""
        return DatasetSummary(
            n_rows=profile.shape[0],
            n_cols=profile.shape[1],
            time_column=profile.time_column,
            target_column=profile.target_column,
            covariates=profile.covariates,
            inferred_frequency=profile.inferred_frequency,
            date_range=profile.date_range,
            missing_pct=profile.missing_pct,
            outlier_pct=profile.outlier_pct,
            mean=profile.mean,
            std=profile.std,
            min=profile.min_val,
            max=profile.max_val,
            adf_statistic=profile.adf_statistic,
            adf_pvalue=profile.adf_pvalue,
            kpss_statistic=profile.kpss_statistic,
            kpss_pvalue=profile.kpss_pvalue,
            has_trend=profile.has_trend,
            has_seasonality=profile.has_seasonality,
            seasonal_period=profile.seasonal_period,
            plot_paths=profile.plot_paths,
        )

    def _detect_seasonality(self, y: np.ndarray) -> tuple[bool, bool, Optional[int]]:
        """Detect trend and seasonality using ACF and STL decomposition."""
        acf_values = acf(y, nlags=min(50, len(y) // 2), fft=True)

        threshold = 2 / np.sqrt(len(y))
        significant_lags = np.where(np.abs(acf_values[1:]) > threshold)[0] + 1

        has_seasonality = len(significant_lags) > 0
        seasonal_period = None

        if has_seasonality and len(y) > 20:
            for period in [7, 12, 24, 30, 365]:
                if len(y) >= 2 * period:
                    try:
                        stl = STL(y, period=period, robust=True)
                        result = stl.fit()
                        seasonal_strength = np.var(result.seasonal) / np.var(y)
                        if seasonal_strength > 0.1:
                            seasonal_period = period
                            break
                    except Exception:
                        continue

        # Trend detection
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        has_trend = abs(slope) > 0.01 * np.std(y)

        return has_trend, has_seasonality, seasonal_period

    def _generate_plots(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        y: np.ndarray,
        output_dir: Path,
    ) -> dict[str, str]:
        """Generate Plotly visualization artifacts."""
        plot_paths = {}

        # 1. Time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[target_col],
            mode="lines", name=target_col,
            line=dict(width=1),
        ))
        fig.update_layout(
            title="Time Series", xaxis_title="Time", yaxis_title=target_col,
            template="plotly_white", height=400,
        )
        path = str(output_dir / "timeseries.html")
        fig.write_html(path)
        plot_paths["timeseries"] = path

        # 2. Distribution plot
        fig = px.histogram(x=y, nbins=50, title="Distribution")
        fig.update_layout(
            xaxis_title=target_col, yaxis_title="Frequency",
            template="plotly_white", height=400,
        )
        path = str(output_dir / "distribution.html")
        fig.write_html(path)
        plot_paths["distribution"] = path

        # 3. ACF plot
        acf_values = acf(y, nlags=min(40, len(y) // 2), fft=True)
        n = len(y)
        fig = go.Figure()
        for i, val in enumerate(acf_values):
            fig.add_trace(go.Scatter(
                x=[i, i], y=[0, val], mode="lines",
                line=dict(color="steelblue", width=2),
                showlegend=False,
            ))
        fig.add_hline(y=2 / np.sqrt(n), line_dash="dash", line_color="red")
        fig.add_hline(y=-2 / np.sqrt(n), line_dash="dash", line_color="red")
        fig.add_hline(y=0, line_color="black", line_width=0.8)
        fig.update_layout(
            title="Autocorrelation Function", xaxis_title="Lag", yaxis_title="ACF",
            template="plotly_white", height=400,
        )
        path = str(output_dir / "acf.html")
        fig.write_html(path)
        plot_paths["acf"] = path

        # 4. Box plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=y, name=target_col, boxpoints="outliers"))
        fig.update_layout(
            title="Box Plot (Outlier Detection)", yaxis_title=target_col,
            template="plotly_white", height=400,
        )
        path = str(output_dir / "boxplot.html")
        fig.write_html(path)
        plot_paths["boxplot"] = path

        return plot_paths
