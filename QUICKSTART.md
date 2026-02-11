# Quick Start Guide

Get TinyTS-Scientist running in 5 minutes!

## Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.10 or higher
- [ ] 8GB RAM (16GB recommended)
- [ ] 4GB GPU (optional, can run on CPU)

## Step 1: Install Ollama

### Linux/macOS
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows
Download from https://ollama.ai/download

### Pull the model
```bash
ollama pull llama3.2:3b
```

**Verify**: Run `ollama list` - you should see `llama3.2:3b`

## Step 2: Install TinyTS-Scientist

```bash
# Clone repository
git clone <repo-url>
cd tiny-agent

# Install with Poetry (recommended)
make setup

# OR install with pip
pip install -r requirements.txt
cp .env.example .env
python examples/generate_sample_data.py
```

## Step 3: Start Ollama

In a separate terminal:
```bash
ollama serve
```

Keep this running!

## Step 4: Run Your First Analysis

```bash
# Using Make
make run-example

# OR using Python directly
poetry run python examples/basic_usage.py
```

## What Happens Next?

The system will:

1. **Load data** (`data/raw/seasonal_sales.csv`)
2. **Profile the data** (statistics, plots)
3. **LLM reasoning** (analyze patterns, suggest models)
4. **Ask for your approval** ⚠️ **INTERACTIVE STEP**
   - Review the LLM's reasoning
   - Edit if needed
   - Approve to continue
5. **Train models** (ARIMA, LightGBM, etc.)
6. **Select best strategy** (single model or ensemble)
7. **Generate forecasts**
8. **Create report**

## Expected Output

After completion, check `outputs/run_XXXXX/`:

```
outputs/run_XXXXX/
├── timeseries.png          # Your data visualization
├── distribution.png        # Distribution plot
├── acf.png                 # Autocorrelation
├── boxplot.png             # Outlier detection
├── final_forecast.png      # Forecast visualization
├── model_comparison.png    # Model comparison
├── approved_reasoning.json # What you approved
└── final_report.md         # Natural language report
```

## Using Your Own Data

```bash
# CLI
poetry run python main.py run YOUR_DATA.csv \
  --time date_column \
  --target value_column \
  --output outputs/my_analysis

# Python API
from tinyts.graph import create_graph

graph = create_graph()
result = graph.run(
    dataset_path="YOUR_DATA.csv",
    time_column="date_column",
    target_column="value_column",
)
```

### Data Requirements

Your CSV/Parquet should have:
- **Time column**: Datetime or parseable date strings
- **Target column**: Numeric values to forecast
- **Optional**: Covariate columns

Example:
```csv
date,sales,temperature,holiday
2023-01-01,100,15,0
2023-01-02,105,16,0
2023-01-03,110,14,1
...
```

## Troubleshooting

### "Connection refused to Ollama"
**Solution**: Make sure Ollama is running: `ollama serve`

### "CUDA out of memory"
**Solution**: Edit `.env` and set `DEVICE=cpu`

### "Module not found"
**Solution**: Activate environment: `poetry shell`

### LLM is slow
**Expected**: First query takes 5-10s on CPU, subsequent queries are faster

## Next Steps

1. **Read the docs**:
   - [README.md](README.md) - Overview
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Design details
   - [SETUP.md](SETUP.md) - Detailed setup

2. **Try different datasets**:
   ```bash
   # Generate more samples
   python examples/generate_sample_data.py
   
   # Try anomaly detection
   poetry run python main.py run data/raw/anomaly_data.csv \
     --time timestamp --target value
   ```

3. **Customize**:
   - Edit prompts in `tinyts/nodes/reasoning_agent.py`
   - Add models in `tinyts/tools/`
   - Adjust hyperparameters in `tinyts/nodes/model_planner.py`

## Common Use Cases

### Forecasting Sales
```bash
poetry run python main.py run sales.csv \
  --time date --target revenue \
  --covariates "marketing_spend,holiday"
```

### Anomaly Detection
```bash
poetry run python main.py run metrics.csv \
  --time timestamp --target cpu_usage
```

### Multi-step Forecast
The system automatically determines horizon based on your data.
Typical: 10% of data length or one seasonal period.

## Getting Help

- **Issues**: Open a GitHub issue
- **Questions**: Check [ARCHITECTURE.md](ARCHITECTURE.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Performance Tips

### For 4GB GPU
- Use `llama3.2:3b` (default)
- Limit neural models
- Reduce dataset size if needed

### For CPU-only
- Set `DEVICE=cpu` in `.env`
- Expect 5-10s per LLM query
- Statistical/tree models work great

### For Large Datasets
- Sample your data first
- Increase `MAX_WORKERS` in `.env`
- Use tree models (faster than neural)

## Success Checklist

After running the example, you should have:
- [x] Ollama running with llama3.2:3b
- [x] Sample data generated
- [x] Successful run with human approval
- [x] Output directory with plots and report
- [x] Understanding of the workflow

**Congratulations! You're ready to use TinyTS-Scientist! 🎉**

