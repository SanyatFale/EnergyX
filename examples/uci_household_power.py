"""
UCI Household Electric Power Consumption Dataset Example

This example demonstrates TinyTS-Scientist on real-world energy consumption data.
Dataset: LD2011_2014.txt from UCI Machine Learning Repository
- 140,257 observations (15-minute intervals, 2011-2014)
- 370 clients (MT_001 to MT_370)
- Values in kW (divide by 4 to get kWh)
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyts.agent import TinyTSAgent


def prepare_uci_data(
    input_file: str = "LD2011_2014.txt",
    client: str = "MT_001",
    sample_size: int = 10000,
    output_file: str = "data/processed/uci_household_power.csv"
):
    """
    Prepare UCI household power data for TinyTS-Scientist.

    Args:
        input_file: Path to LD2011_2014.txt
        client: Which client to analyze (MT_001 to MT_370)
        sample_size: Number of recent observations to use (None for all)
        output_file: Where to save processed data
    """
    print(f"\n{'='*80}")
    print(f"Preparing UCI Household Power Data")
    print(f"{'='*80}\n")

    print(f"Loading data from {input_file}...")

    # Read the data
    df = pd.read_csv(
        input_file,
        sep=";",
        parse_dates=[0],
        index_col=0,
        decimal=","  # European decimal format
    )

    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} clients")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Frequency: {pd.infer_freq(df.index[:100])}")

    # Select client
    if client not in df.columns:
        print(f"\n✗ Client {client} not found!")
        print(f"  Available clients: {', '.join(df.columns[:10])}...")
        return None

    # Extract single client
    client_data = df[[client]].copy()
    client_data.columns = ['power_kw']

    # Remove zeros (clients created after 2011)
    client_data = client_data[client_data['power_kw'] > 0]

    print(f"\n✓ Selected client: {client}")
    print(f"  Non-zero observations: {len(client_data):,}")
    print(f"  Mean: {client_data['power_kw'].mean():.2f} kW")
    print(f"  Std: {client_data['power_kw'].std():.2f} kW")
    print(f"  Min: {client_data['power_kw'].min():.2f} kW")
    print(f"  Max: {client_data['power_kw'].max():.2f} kW")

    # Sample if needed
    if sample_size and len(client_data) > sample_size:
        print(f"\n✓ Sampling last {sample_size:,} observations...")
        client_data = client_data.tail(sample_size)

    # Reset index to make datetime a column
    client_data = client_data.reset_index()
    client_data.columns = ['timestamp', 'power_kw']

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client_data.to_csv(output_path, index=False)

    print(f"\n✓ Saved to {output_path}")
    print(f"  Final dataset: {len(client_data):,} rows")

    return str(output_path)


def run_analysis(data_path: str):
    """Run TinyTS-Scientist on the prepared data."""

    print(f"\n{'='*80}")
    print(f"Running TinyTS-Scientist Analysis")
    print(f"{'='*80}\n")

    agent = TinyTSAgent(
        dataset_path=data_path,
        time_column="timestamp",
        target_column="power_kw",
    )

    task_plan = agent.plan("Forecast power consumption for next 96 steps with explanation")
    print(f"Plan: {task_plan.task_type} | horizon={task_plan.horizon}")

    result = agent.execute(task_plan)
    return result


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"UCI Household Electric Power Consumption - TinyTS-Scientist")
    print(f"{'='*80}\n")

    # Configuration
    CLIENT = "MT_200"  # Change this to analyze different clients
    SAMPLE_SIZE = 5000  # Use last 5000 observations (~52 days at 15-min intervals)

    print(f"Configuration:")
    print(f"  Client: {CLIENT}")
    print(f"  Sample size: {SAMPLE_SIZE:,} observations")
    print(f"  Expected duration: ~52 days of 15-minute data")

    # Prepare data
    data_path = prepare_uci_data(
        input_file="LD2011_2014.txt",
        client=CLIENT,
        sample_size=SAMPLE_SIZE,
    )

    if data_path is None:
        print("\n✗ Data preparation failed!")
        sys.exit(1)

    # Run analysis
    result = run_analysis(data_path)

    # Print summary
    print(f"\n{'='*80}")
    print(f"✓ Analysis Complete!")
    print(f"{'='*80}\n")

    print(f"Output Directory: {result['output_dir']}")

    if result.get("response"):
        print(f"\nAgent Summary:\n{result['response']}")

    report = result["session"].get("report")
    if report:
        print(f"\nReport: {result['output_dir']}/final_report.md")

    print(f"\n{'='*80}\n")
