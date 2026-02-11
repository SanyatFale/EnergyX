#!/usr/bin/env python3
"""
Universal Time Series Analysis Script

Automatically analyzes any time series dataset and runs TinyTS-Scientist.
Handles CSV, Parquet, and other common formats.
Automatically detects datetime and numeric columns.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from tinyts.agent import TinyTSAgent


def detect_datetime_column(df: pd.DataFrame) -> str:
    """Automatically detect the datetime column."""

    # Check for common datetime column names
    common_names = ['date', 'datetime', 'timestamp', 'time', 'ds', 'Date', 'DateTime', 'Timestamp']
    for name in common_names:
        if name in df.columns:
            return name

    # Check for columns that can be parsed as datetime
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(100))
                return col
            except:
                pass
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # If first column looks like an index, use it
    if df.index.name:
        return df.index.name

    raise ValueError("Could not automatically detect datetime column. Please specify with --time")


def detect_target_column(df: pd.DataFrame, time_col: str) -> str:
    """Automatically detect the target column (first numeric column)."""

    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    numeric_cols = [c for c in numeric_cols if c != time_col]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found. Please check your data.")

    return numeric_cols[0]


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various formats."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Detect file format
    suffix = path.suffix.lower()

    if suffix in ['.csv', '.txt', '.dat']:
        # Try different separators
        for sep in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(file_path, sep=sep, nrows=5, decimal=',')
                if len(df.columns) > 1:
                    df = pd.read_csv(file_path, sep=sep, decimal=',')
                    print(f"✓ Loaded {suffix} file with separator '{sep}'")
                    return df
            except:
                pass
            # Try with standard decimal point
            try:
                df = pd.read_csv(file_path, sep=sep, nrows=5)
                if len(df.columns) > 1:
                    df = pd.read_csv(file_path, sep=sep)
                    print(f"✓ Loaded {suffix} file with separator '{sep}'")
                    return df
            except:
                continue
        raise ValueError(f"Could not parse {suffix} file. Please check the format.")

    elif suffix == '.parquet':
        df = pd.read_parquet(file_path)
        print(f"✓ Loaded Parquet file")
        return df

    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        print(f"✓ Loaded Excel file")
        return df

    elif suffix == '.json':
        df = pd.read_json(file_path)
        print(f"✓ Loaded JSON file")
        return df

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Time Series Analysis with TinyTS-Scientist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect everything
  python run_analysis.py data.csv

  # Specify columns
  python run_analysis.py data.csv --time timestamp --target sales

  # With covariates
  python run_analysis.py data.csv --time date --target revenue --covariates "temp,holiday"

  # Sample large datasets
  python run_analysis.py large_data.csv --sample 10000
        """
    )

    parser.add_argument('file', help='Path to time series data file')
    parser.add_argument('--time', help='Name of datetime column (auto-detected if not specified)')
    parser.add_argument('--target', help='Name of target column (auto-detected if not specified)')
    parser.add_argument('--covariates', help='Comma-separated list of covariate columns')
    parser.add_argument('--sample', type=int, help='Sample last N observations (for large datasets)')
    parser.add_argument('--output', help='Output directory (auto-generated if not specified)')
    parser.add_argument('--query', help='Natural-language query (e.g. "forecast next 7 days")')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"TinyTS-Scientist - Universal Time Series Analysis")
    print(f"{'='*80}\n")

    # Load data
    print(f"Loading data from: {args.file}")
    df = load_data(args.file)
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns\n")

    # Detect columns
    time_col = args.time or detect_datetime_column(df)
    print(f"✓ Datetime column: {time_col}")

    target_col = args.target or detect_target_column(df, time_col)
    print(f"✓ Target column: {target_col}")

    covariates = args.covariates.split(',') if args.covariates else None
    if covariates:
        print(f"✓ Covariates: {', '.join(covariates)}")

    # Sample if needed
    if args.sample and len(df) > args.sample:
        print(f"\n✓ Sampling last {args.sample:,} observations...")
        df = df.tail(args.sample)

    # Save processed data
    temp_file = Path("data/processed/temp_analysis.csv")
    temp_file.parent.mkdir(parents=True, exist_ok=True)

    cols_to_save = [time_col, target_col]
    if covariates:
        cols_to_save.extend(covariates)

    df[cols_to_save].to_csv(temp_file, index=False)
    print(f"\n✓ Prepared data: {len(df):,} rows\n")

    # Run analysis
    print(f"{'='*80}")
    print(f"Running TinyTS-Scientist...")
    print(f"{'='*80}\n")

    user_query = args.query or f"Forecast {target_col}"

    agent = TinyTSAgent(
        dataset_path=str(temp_file),
        time_column=time_col,
        target_column=target_col,
        output_dir=args.output,
        feature_columns=covariates,
    )

    task_plan = agent.plan(user_query)
    print(f"Plan: {task_plan.task_type} | horizon={task_plan.horizon} | models={task_plan.models_included}")

    result = agent.execute(task_plan)

    # Print results
    print(f"\n{'='*80}")
    print(f"✓ Analysis Complete!")
    print(f"{'='*80}\n")
    print(f"Output: {result['output_dir']}")

    if result.get("response"):
        print(f"\nAgent Summary:\n{result['response']}")

    report = result["session"].get("report")
    if report:
        print(f"\nReport preview:")
        print(report[:500] + "...\n")
        print(f"Full report: {result['output_dir']}/final_report.md")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
