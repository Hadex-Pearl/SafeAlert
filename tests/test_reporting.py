"""Unit tests for SafeAlert reporting tables and charts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from scripts.metrics import compute_metrics
from scripts.reporting import (
    failure_mode_table,
    metrics_summary_table,
    plot_metric_bar_chart,
    run_summary_table,
    write_report_artifacts,
)
from tests.test_metrics import scored_frame


class ReportingTests(unittest.TestCase):
    def test_metrics_summary_table_contains_core_metrics(self) -> None:
        summary = metrics_summary_table(compute_metrics(scored_frame()))

        self.assertEqual(summary["metric"].tolist(), ["Refusal Rate", "TPR", "FNR", "FPR"])
        self.assertAlmostEqual(summary.loc[summary["metric"] == "Refusal Rate", "value"].item(), 2 / 3)
        self.assertAlmostEqual(summary.loc[summary["metric"] == "TPR", "value"].item(), 1 / 3)
        self.assertAlmostEqual(summary.loc[summary["metric"] == "FNR", "value"].item(), 2 / 3)
        self.assertAlmostEqual(summary.loc[summary["metric"] == "FPR", "value"].item(), 2 / 3)

    def test_run_summary_table_contains_counts(self) -> None:
        summary = run_summary_table(compute_metrics(scored_frame()))

        self.assertEqual(summary["section"].tolist(), ["all", "generation", "classification"])
        self.assertEqual(summary.loc[summary["section"] == "all", "total"].item(), 12)
        self.assertEqual(summary.loc[summary["section"] == "generation", "excluded_error"].item(), 1)
        self.assertEqual(summary.loc[summary["section"] == "classification", "excluded_error"].item(), 1)

    def test_failure_mode_table_groups_incorrect_classifications(self) -> None:
        failures = failure_mode_table(scored_frame())

        self.assertEqual(failures["error_type"].tolist(), ["false_negative", "false_positive"])
        self.assertEqual(failures["count"].tolist(), [2, 2])
        self.assertEqual(failures["percent"].tolist(), [50.0, 50.0])

    def test_failure_mode_table_handles_no_failures(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "prompt_type": "classification",
                    "api_status": "success",
                    "score": 1,
                    "ground_truth_label": "safe",
                    "label_assigned": "safe",
                    "error_type": "",
                }
            ]
        )

        failures = failure_mode_table(scored)

        self.assertEqual(failures.columns.tolist(), ["error_type", "count", "percent"])
        self.assertTrue(failures.empty)

    def test_metric_bar_chart_writes_png(self) -> None:
        summary = metrics_summary_table(compute_metrics(scored_frame()))
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.png"

            plot_metric_bar_chart(summary, output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_write_report_artifacts_writes_tables_and_chart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifacts = write_report_artifacts(
                scored=scored_frame(),
                summary_csv=output_dir / "summary.csv",
                failure_modes_csv=output_dir / "failures.csv",
                chart_png=output_dir / "metrics.png",
            )

            self.assertEqual(set(artifacts), {"metrics_summary", "failure_modes"})
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "failures.csv").exists())
            self.assertTrue((output_dir / "metrics.png").exists())


if __name__ == "__main__":
    unittest.main()
