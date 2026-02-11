"""Basic usage example for TinyTS-Scientist."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinyts.agent import TinyTSAgent


def main():
    """Run a basic forecasting example."""

    # Path to sample data
    data_path = "data/raw/seasonal_sales.csv"

    # Check if data exists
    if not Path(data_path).exists():
        print(f"Sample data not found at {data_path}")
        print("Run: python examples/generate_sample_data.py")
        return

    print("="*80)
    print("TinyTS-Scientist - Basic Usage Example")
    print("="*80)
    print()
    print("This example demonstrates:")
    print("  1. Loading a time series dataset")
    print("  2. Automatic data profiling")
    print("  3. LLM-based query understanding")
    print("  4. Automated model training and selection")
    print("  5. Final forecast generation")
    print("  6. Natural language report")
    print()
    print("="*80)
    print()

    # Create agent
    agent = TinyTSAgent(
        dataset_path=data_path,
        time_column="date",
        target_column="sales",
        feature_columns=["day_of_week", "month"],
    )

    # Plan
    task_plan = agent.plan("Forecast sales for next 14 days with explanation")

    print(f"Plan: {task_plan.task_type} | horizon={task_plan.horizon}")
    print(f"Models: {task_plan.models_included}")
    print(f"Explanation: {task_plan.needs_explanation}")
    print()

    # Execute
    result = agent.execute(task_plan)
    session = result["session"]

    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nOutput Directory: {result['output_dir']}")

    profile = session.get("profile")
    if profile:
        print(f"\nDataset: {profile.shape[0]} observations")
        print(f"Frequency: {profile.inferred_frequency}")
        print(f"Seasonality: {profile.has_seasonality}")
        print(f"Trend: {profile.has_trend}")

    model_results = session.get("model_results", {})
    if model_results:
        print(f"\nModels Trained: {len(model_results)}")
        print("\nModel Performance (MAPE):")
        for name, res in sorted(model_results.items(), key=lambda x: x[1]["mean_mape"]):
            print(f"  - {name}: {res['mean_mape']:.2f}% (+-{res['std_mape']:.2f}%)")

    strat = session.get("ensemble_strategy")
    if strat:
        print(f"\nStrategy: {strat['strategy_type']}")
        if strat.get("model_weights"):
            print("Model Weights:")
            for name, weight in strat["model_weights"].items():
                print(f"  - {name}: {weight:.3f}")

    preds = session.get("final_predictions")
    if preds:
        print(f"\nGenerated {len(preds)} predictions")
        print(f"First 5: {[round(v, 2) for v in preds[:5]]}")

    for name, exp in session.get("model_explanations", {}).items():
        fi = exp.get("feature_importance", {})
        if fi:
            print(f"\nFeature Importance ({name}):")
            for k, v in sorted(fi.items(), key=lambda x: -abs(x[1]))[:5]:
                print(f"  - {k}: {v:.4f}")

    if result.get("response"):
        print(f"\nAgent Summary:\n{result['response']}")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print(f"View full report: {result['output_dir']}/final_report.md")
    print("="*80)


if __name__ == "__main__":
    main()
