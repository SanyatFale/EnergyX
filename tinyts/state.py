"""State management for LangGraph workflow."""

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    """Summary statistics from data profiling."""

    n_rows: int
    n_cols: int
    time_column: str
    target_column: str
    covariates: List[str]

    # Frequency and temporal properties
    inferred_frequency: Optional[str] = None
    date_range: tuple[str, str]

    # Statistical properties
    missing_pct: float
    outlier_pct: float
    mean: float
    std: float
    min: float
    max: float

    # Stationarity tests
    adf_statistic: float
    adf_pvalue: float
    kpss_statistic: float
    kpss_pvalue: float

    # Seasonality
    has_trend: bool
    has_seasonality: bool
    seasonal_period: Optional[int] = None

    # Visualization paths
    plot_paths: Dict[str, str] = Field(default_factory=dict)


class DataProfile(BaseModel):
    """Enhanced dataset profile for UI display.

    Superset of DatasetSummary — adds column-level info, head rows,
    and column role suggestions for the Streamlit interface.
    """

    # Column-level information
    columns: List[Dict[str, Any]] = Field(default_factory=list)
    # [{name, dtype, missing_pct, unique_count, sample_values}]

    head_rows: List[Dict[str, Any]] = Field(default_factory=list)
    # First 10 rows as list of dicts

    shape: tuple[int, int]

    # User-provided description
    description: Optional[str] = None

    # Column role suggestions (auto-detected)
    datetime_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)

    # All DatasetSummary fields
    time_column: str
    target_column: str
    covariates: List[str] = Field(default_factory=list)
    inferred_frequency: Optional[str] = None
    date_range: tuple[str, str] = ("", "")
    missing_pct: float = 0.0
    outlier_pct: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    adf_statistic: float = 0.0
    adf_pvalue: float = 1.0
    kpss_statistic: float = 0.0
    kpss_pvalue: float = 1.0
    has_trend: bool = False
    has_seasonality: bool = False
    seasonal_period: Optional[int] = None
    plot_paths: Dict[str, str] = Field(default_factory=dict)


class UserTaskPlan(BaseModel):
    """What the user wants done, extracted from their natural language query.

    Produced by QueryUnderstandingNode, editable by user in Streamlit UI.
    """
    model_config = {"extra": "ignore"}  # Tolerate unknown fields from LLM JSON

    user_query: str
    task_type: str  # "forecast", "anomaly", "both"
    is_multivariate: bool = False
    target_column: str
    feature_columns: List[str] = Field(default_factory=list)
    horizon: Optional[int] = None
    models_included: List[str] = Field(default_factory=list)
    models_excluded: List[str] = Field(default_factory=list)
    needs_cv: bool = True
    needs_explanation: bool = False
    needs_report: bool = False
    needs_plots: bool = True
    explanation: str = ""  # LLM's interpretation of the query

    # Counterfactual analysis
    counterfactual_type: Optional[str] = None  # "forward" | "inverse" | None
    counterfactual_changes: Dict[str, float] = Field(default_factory=dict)
    # Forward: {"air_temperature": -5} means "drop temp by 5"
    counterfactual_target_value: Optional[float] = None
    # Inverse: desired target value
    counterfactual_constraints: Dict[str, List[float]] = Field(default_factory=dict)
    # Inverse: {"air_temperature": [0, 45]} means min=0, max=45


class ReasoningOutput(BaseModel):
    """Structured output from reasoning agent (kept for backward compatibility)."""

    problem_type: str  # "forecasting", "anomaly", "both"
    seasonality_strength: str  # "none", "weak", "moderate", "strong"
    non_stationarity_risk: str  # "low", "medium", "high"

    candidate_models: List[str]
    cv_strategy: str
    ensemble_recommended: bool

    reasoning: str  # Natural language explanation
    assumptions: List[str]
    limitations: List[str]


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    model_name: str
    model_family: str  # "statistical", "tree", "neural"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    search_space: Optional[Dict[str, List[Any]]] = None


class ModelPlan(BaseModel):
    """Complete model training plan."""

    models: List[ModelConfig]
    cv_folds: int
    cv_horizon: int
    metric: str = "mape"
    is_multivariate: bool = False
    feature_columns: List[str] = Field(default_factory=list)


class ModelResult(BaseModel):
    """Results from a single model training."""

    model_name: str
    cv_scores: List[float]
    mean_score: float
    std_score: float
    best_hyperparameters: Dict[str, Any]
    training_time: float


class EnsembleStrategy(BaseModel):
    """Ensemble configuration."""

    strategy_type: str  # "best_model", "weighted", "median"
    model_weights: Optional[Dict[str, float]] = None
    selected_models: List[str]
    llm_recommendation: str = ""  # LLM's suggestion text


class ExplainabilityResult(BaseModel):
    """Results from the explainability suite.

    Ported from OwnSolarCast explain.py — holds all computed metrics
    plus the LLM-generated explanation.
    """

    statistical_summary: Optional[Dict[str, Any]] = None
    decomposition_metrics: Optional[Dict[str, Any]] = None
    lag_contributions: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    shap_summary: Optional[Dict[str, float]] = None
    feature_correlations: Optional[Dict[str, float]] = None
    anomaly_method_agreement: Optional[Dict[str, Any]] = None
    llm_explanation: str = ""
    plot_paths: Dict[str, str] = Field(default_factory=dict)


class AgentState(TypedDict):
    """Complete state for the LangGraph workflow."""

    # Input
    dataset_path: str
    time_column: str
    target_column: str
    covariates: Optional[List[str]]
    user_query: Optional[str]

    # Data Profiler outputs
    dataset_summary: Optional[DatasetSummary]
    data_profile: Optional[DataProfile]

    # Query Understanding outputs
    user_task_plan: Optional[UserTaskPlan]

    # Reasoning Agent outputs (backward compat for CLI mode)
    reasoning_output: Optional[ReasoningOutput]
    approved_reasoning: Optional[ReasoningOutput]

    # Model Planner outputs
    model_plan: Optional[ModelPlan]

    # Training outputs
    model_results: Optional[List[ModelResult]]

    # Strategy Selector outputs
    ensemble_strategy: Optional[EnsembleStrategy]

    # Execution outputs
    predictions: Optional[List[float]]
    prediction_intervals: Optional[List[tuple[float, float]]]
    per_model_predictions: Optional[Dict[str, List[float]]]
    per_model_metrics: Optional[Dict[str, Dict[str, float]]]

    # Anomaly detection outputs
    anomaly_results: Optional[Dict[str, Any]]

    # Explainability
    explainability_result: Optional[ExplainabilityResult]

    # Report
    final_report: Optional[str]

    # Metadata
    run_id: str
    output_dir: str
