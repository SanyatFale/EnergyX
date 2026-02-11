# TinyTS-Scientist Examples

This directory contains example scripts and sample datasets for TinyTS-Scientist.

## Quick Start

### 1. Generate Sample Data

```bash
python examples/generate_sample_data.py
```

This creates three sample datasets in `data/raw/`:

- **seasonal_sales.csv** - Sales data with weekly seasonality
  - 365 daily observations
  - Weekly seasonal pattern
  - Linear trend
  - Some outliers

- **anomaly_data.csv** - Time series with anomalies
  - 500 hourly observations
  - ~5% anomalies (extreme values)
  - Useful for testing anomaly detection

- **trend_data.csv** - Data with exponential trend
  - 200 weekly observations
  - Strong exponential trend
  - No seasonality
  - Useful for testing trend models

### 2. Run Basic Example

```bash
python examples/basic_usage.py
```

This demonstrates the complete workflow:
1. Load seasonal sales data
2. Profile the data
3. LLM reasoning
4. Human approval (interactive)
5. Model training
6. Forecast generation
7. Report creation

## Example Outputs

After running `basic_usage.py`, check `outputs/run_XXXXX/`:

```
outputs/run_XXXXX/
├── timeseries.png          # Time series visualization
├── distribution.png        # Value distribution
├── acf.png                 # Autocorrelation function
├── boxplot.png             # Outlier detection
├── final_forecast.png      # Final forecast
├── model_comparison.png    # Compare all models
├── approved_reasoning.json # Your approved plan
└── final_report.md         # Natural language report
```

## Using Your Own Data

### CSV Format

Your CSV should have at minimum:
- A time/date column
- A numeric target column

Example:
```csv
date,sales,temperature,holiday
2023-01-01,100,15,0
2023-01-02,105,16,0
2023-01-03,110,14,1
```

### Running Analysis

```python
from tinyts.graph import create_graph

graph = create_graph()
result = graph.run(
    dataset_path="your_data.csv",
    time_column="date",
    target_column="sales",
    covariates=["temperature", "holiday"],  # Optional
)
```

## Sample Datasets Details

### seasonal_sales.csv

**Use case**: Retail sales forecasting

**Properties**:
- Frequency: Daily
- Length: 365 days
- Seasonality: Weekly (7-day period)
- Trend: Positive linear
- Noise: Moderate
- Outliers: ~2%

**Expected models**: SeasonalNaive, ARIMA, LightGBM

### anomaly_data.csv

**Use case**: System monitoring, anomaly detection

**Properties**:
- Frequency: Hourly
- Length: 500 hours
- Anomalies: ~5% (extreme values)
- Base pattern: Normal distribution

**Expected models**: IsolationForest, Statistical

### trend_data.csv

**Use case**: Growth metrics, KPI tracking

**Properties**:
- Frequency: Weekly
- Length: 200 weeks
- Trend: Exponential
- Seasonality: None
- Noise: Low

**Expected models**: ARIMA, ETS, LightGBM

## Advanced Examples

### Custom Model Selection

Edit the reasoning during human approval to select specific models:

```
Candidate Models: ARIMA, LightGBM, N-BEATS
```

### Different Metrics

The system uses MAPE by default. To use different metrics, modify `ModelPlan.metric` in the code.

### Longer Forecasts

The system automatically determines horizon (typically 10% of data or one seasonal period). For custom horizons, modify `ModelPlannerNode._infer_horizon()`.

## Troubleshooting

### "Dataset not found"

Make sure you've run `generate_sample_data.py` first:
```bash
python examples/generate_sample_data.py
```

### "Ollama connection error"

Start Ollama in a separate terminal:
```bash
ollama serve
```

### "Module not found"

Activate the Poetry environment:
```bash
poetry shell
```

## Next Steps

1. **Try different datasets**: Use your own CSV files
2. **Experiment with approval**: Edit model selection during human approval
3. **Compare strategies**: Try best model vs ensemble
4. **Customize prompts**: Edit `tinyts/nodes/reasoning_agent.py`
5. **Add models**: Create new tools in `tinyts/tools/`

## Contributing Examples

To contribute new examples:

1. Create a new Python file in `examples/`
2. Add documentation in this README
3. Ensure it works with `poetry run python examples/your_example.py`
4. Submit a pull request

## Support

For issues with examples:
- Check the main [README.md](../README.md)
- Review [QUICKSTART.md](../QUICKSTART.md)
- Open a GitHub issue

