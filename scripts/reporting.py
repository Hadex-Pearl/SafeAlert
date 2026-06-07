"""Summary tables and visualisations for SafeAlert metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from metrics import CLASSIFICATION_TYPE, ERROR_STATUS, compute_metrics, load_scored_csv
except ModuleNotFoundError:
    from scripts.metrics import CLASSIFICATION_TYPE, ERROR_STATUS, compute_metrics, load_scored_csv


CORE_METRICS = (
    ("Refusal Rate", ("generation", "refusal_rate")),
    ("TPR", ("classification", "TPR")),
    ("FNR", ("classification", "FNR")),
    ("FPR", ("classification", "FPR")),
)


def metrics_summary_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Build a compact table of the core SafeAlert metrics."""
    rows = []
    for metric_name, path in CORE_METRICS:
        value = _nested_get(metrics, path)
        rows.append(
            {
                "metric": metric_name,
                "value": value,
                "percent": None if value is None else round(value * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def run_summary_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Build a high-level run count summary table."""
    generation = metrics["generation"]
    classification = metrics["classification"]
    return pd.DataFrame(
        [
            {
                "section": "all",
                "total": metrics["total_prompts"],
                "excluded_error": metrics["excluded_error"],
            },
            {
                "section": "generation",
                "total": generation["total"],
                "excluded_error": generation["excluded"],
            },
            {
                "section": "classification",
                "total": classification["total"],
                "excluded_error": classification["excluded"],
            },
        ]
    )


def failure_mode_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarize incorrect classification rows by failure mode."""
    required_columns = {"prompt_type", "api_status", "score", "error_type"}
    missing_columns = sorted(required_columns - set(scored.columns))
    if missing_columns:
        raise ValueError(f"Missing required scored column(s): {', '.join(missing_columns)}")

    failures = scored[
        (scored["prompt_type"] == CLASSIFICATION_TYPE)
        & (scored["api_status"] != ERROR_STATUS)
        & (pd.to_numeric(scored["score"], errors="coerce") == 0)
    ].copy()
    failures["error_type"] = failures["error_type"].replace("", "unspecified")

    if failures.empty:
        return pd.DataFrame(columns=["error_type", "count", "percent"])

    counts = failures["error_type"].value_counts().rename_axis("error_type").reset_index(name="count")
    counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return counts.sort_values(["count", "error_type"], ascending=[False, True]).reset_index(drop=True)


def plot_metric_bar_chart(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Write a bar chart for the core metric summary table."""
    chart_data = summary.dropna(subset=["value"]).copy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=chart_data, x="metric", y="value", ax=axis, color="#4C78A8")
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_ylabel("Rate")
    axis.set_title("SafeAlert Core Metrics")

    for patch in axis.patches:
        height = patch.get_height()
        axis.annotate(
            f"{height:.1%}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def write_report_artifacts(
    scored: pd.DataFrame,
    summary_csv: str | Path | None = None,
    failure_modes_csv: str | Path | None = None,
    chart_png: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute report tables and optionally write them and the bar chart to disk."""
    metrics = compute_metrics(scored)
    metric_summary = metrics_summary_table(metrics)
    failures = failure_mode_table(scored)

    if summary_csv:
        _write_csv(metric_summary, summary_csv)
    if failure_modes_csv:
        _write_csv(failures, failure_modes_csv)
    if chart_png:
        plot_metric_bar_chart(metric_summary, chart_png)

    return {"metrics_summary": metric_summary, "failure_modes": failures}


def _nested_get(values: dict[str, Any], path: tuple[str, str]) -> Any:
    current: Any = values
    for key in path:
        current = current[key]
    return current


def _write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def main() -> None:
    """Generate summary tables and a metric bar chart from a scored CSV."""
    parser = argparse.ArgumentParser(description="Build SafeAlert summary tables and visualisations.")
    parser.add_argument("scored_csv", help="Path to a manually scored SafeAlert CSV file.")
    parser.add_argument("--summary-csv", help="Optional path for the core metrics summary CSV.")
    parser.add_argument("--failure-modes-csv", help="Optional path for the failure mode table CSV.")
    parser.add_argument("--chart-png", help="Optional path for the core metrics bar chart PNG.")
    args = parser.parse_args()

    artifacts = write_report_artifacts(
        scored=load_scored_csv(args.scored_csv),
        summary_csv=args.summary_csv,
        failure_modes_csv=args.failure_modes_csv,
        chart_png=args.chart_png,
    )
    print("Metrics summary:")
    print(artifacts["metrics_summary"].to_string(index=False))
    print("\nFailure modes:")
    print(artifacts["failure_modes"].to_string(index=False))


if __name__ == "__main__":
    main()
