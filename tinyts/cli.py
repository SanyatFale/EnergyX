"""Command-line interface for TinyTS-Scientist."""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    name="tinyts",
    help="TinyTS-Scientist: A Local, Human-in-the-Loop Agentic Time-Series System",
)

console = Console()


def setup_logging(level: str = "INFO"):
    """Setup logging with rich handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Path to dataset (CSV or Parquet)"),
    time_column: str = typer.Option(..., "--time", "-t", help="Name of time column"),
    target_column: str = typer.Option(..., "--target", "-y", help="Name of target column"),
    covariates: Optional[str] = typer.Option(None, "--covariates", "-c", help="Comma-separated covariate columns"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Natural-language query (e.g. 'forecast next 7 days')"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    auto_approve: bool = typer.Option(False, "--yes", "-Y", help="Auto-approve the plan without prompting"),
):
    """Run TinyTS-Scientist workflow on a dataset.

    Example:
        tinyts run data.csv --time date --target sales --query "forecast next 7 days"
    """
    setup_logging(log_level)

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    feature_columns = None
    if covariates:
        feature_columns = [c.strip() for c in covariates.split(",")]

    user_query = query or f"Forecast {target_column}"

    try:
        from tinyts.agent import TinyTSAgent

        agent = TinyTSAgent(
            dataset_path=str(dataset_path),
            time_column=time_column,
            target_column=target_column,
            output_dir=output_dir,
            feature_columns=feature_columns,
        )

        # Phase 1: Plan
        console.print("\n[bold]Phase 1: Profiling & Planning[/bold]")
        task_plan = agent.plan(user_query)

        # Show profile
        profile = agent.session.get("profile")
        if profile:
            console.print(f"  Dataset: {profile.shape[0]} rows x {profile.shape[1]} cols")
            console.print(f"  Frequency: {profile.inferred_frequency or 'Unknown'}")
            if profile.date_range:
                console.print(f"  Date range: {profile.date_range[0]} to {profile.date_range[1]}")

        # Show plan
        console.print(f"\n[bold]Plan:[/bold]")
        console.print(f"  Task: {task_plan.task_type}")
        console.print(f"  Horizon: {task_plan.horizon}")
        console.print(f"  Multivariate: {task_plan.is_multivariate}")
        console.print(f"  Models: {task_plan.models_included or 'default'}")
        console.print(f"  Explanation: {task_plan.needs_explanation}")
        console.print(f"  Report: {task_plan.needs_report}")

        if task_plan.reasoning:
            console.print(f"  Reasoning: {task_plan.reasoning}")

        # Human approval
        if not auto_approve:
            console.print()
            approved = typer.confirm("Approve this plan?", default=True)
            if not approved:
                console.print("[yellow]Plan rejected. Exiting.[/yellow]")
                raise typer.Exit(0)

        # Phase 2: Execute
        console.print("\n[bold]Phase 2: Executing[/bold]")

        def _on_tool(name, args, result_str):
            try:
                r = json.loads(result_str)
                if name == "train_forecast_model" or name == "train_and_explain_forecast":
                    console.print(
                        f"  [green]{r.get('model_name', name)}[/green]: "
                        f"MAPE={r.get('mean_mape', '?')}% "
                        f"(+/-{r.get('std_mape', '?')}%)"
                    )
                elif name == "select_ensemble_strategy":
                    console.print(
                        f"  [blue]Strategy:[/blue] {r.get('strategy_type', '?')} — "
                        f"{', '.join(r.get('selected_models', []))}"
                    )
                elif name == "detect_anomalies":
                    console.print(f"  [yellow]Anomalies:[/yellow] {r.get('n_anomalies', 0)} found")
                elif name == "generate_report":
                    console.print("  [green]Report generated[/green]")
                else:
                    console.print(f"  Tool: {name} — done")
            except Exception:
                console.print(f"  Tool: {name} — done")

        result = agent.execute(task_plan, on_tool_call=_on_tool)
        session = result["session"]
        out_dir = result["output_dir"]

        console.print(f"\n[bold green]Analysis Complete![/bold green]")
        console.print(f"Output directory: {out_dir}")

        # Show agent summary
        agent_response = result.get("response", "")
        if agent_response:
            console.print(f"\n[bold]Agent Summary:[/bold]")
            console.print(agent_response)

        # Show report preview
        report = session.get("report")
        if report:
            console.print(f"\n[bold]Report Preview:[/bold]")
            console.print("-" * 80)
            console.print(report[:500] + "...\n")
            console.print(f"Full report: {out_dir}/final_report.md")

        # Show explainability
        for name, exp in session.get("model_explanations", {}).items():
            fi = exp.get("feature_importance", {})
            if fi:
                console.print(f"\n[bold]Feature Importance ({name}):[/bold]")
                for k, v in sorted(fi.items(), key=lambda x: -abs(x[1]))[:10]:
                    console.print(f"  {k}: {v:.4f}")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from tinyts import __version__
    console.print(f"TinyTS-Scientist version {__version__}")


if __name__ == "__main__":
    app()
