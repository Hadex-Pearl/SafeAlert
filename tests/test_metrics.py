"""Unit tests for SafeAlert metric formulas."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.compute_metrics import derive_metric_outcome
from scripts.metrics import (
    compute_metrics,
    false_negative_rate,
    false_positive_rate,
    refusal_rate,
    true_positive_rate,
)


def scored_frame() -> pd.DataFrame:
    """Create a compact scored dataset with generation, classification, and error rows."""
    return pd.DataFrame(
        [
            {
                "prompt_type": "generation",
                "api_status": "success",
                "score": 1,
                "ground_truth_label": "refuse",
                "label_assigned": "",
                "error_type": "",
            },
            {
                "prompt_type": "generation",
                "api_status": "empty_response",
                "score": 1,
                "ground_truth_label": "refuse",
                "label_assigned": "",
                "error_type": "",
            },
            {
                "prompt_type": "generation",
                "api_status": "success",
                "score": 0,
                "ground_truth_label": "refuse",
                "label_assigned": "",
                "error_type": "",
            },
            {
                "prompt_type": "generation",
                "api_status": "error",
                "score": None,
                "ground_truth_label": "refuse",
                "label_assigned": "",
                "error_type": "",
            },
            {
                "prompt_type": "classification",
                "api_status": "success",
                "score": 1,
                "ground_truth_label": "scam",
                "label_assigned": "scam",
                "error_type": "",
            },
            {
                "prompt_type": "classification",
                "api_status": "success",
                "score": 0,
                "ground_truth_label": "scam",
                "label_assigned": "safe",
                "error_type": "false_negative",
            },
            {
                "prompt_type": "classification",
                "api_status": "success",
                "score": 0,
                "ground_truth_label": "scam",
                "label_assigned": "suspicious",
                "error_type": "false_negative",
            },
            {
                "prompt_type": "classification",
                "api_status": "error",
                "score": None,
                "ground_truth_label": "scam",
                "label_assigned": "",
                "error_type": "",
            },
            {
                "prompt_type": "classification",
                "api_status": "success",
                "score": 1,
                "ground_truth_label": "safe",
                "label_assigned": "safe",
                "error_type": "",
            },
            {
                "prompt_type": "classification",
                "api_status": "success",
                "score": 0,
                "ground_truth_label": "safe",
                "label_assigned": "scam",
                "error_type": "false_positive",
            },
            {
                "prompt_type": "classification",
                "api_status": "success",
                "score": 0,
                "ground_truth_label": "safe",
                "label_assigned": "suspicious",
                "error_type": "false_positive",
            },
            {
                "prompt_type": "classification",
                "api_status": "success",
                "score": 1,
                "ground_truth_label": "suspicious",
                "label_assigned": "suspicious",
                "error_type": "",
            },
        ]
    )


class MetricFormulaTests(unittest.TestCase):
    def test_refusal_rate_excludes_generation_errors(self) -> None:
        self.assertAlmostEqual(refusal_rate(scored_frame()), 2 / 3)

    def test_true_positive_rate_uses_scam_denominator(self) -> None:
        self.assertAlmostEqual(true_positive_rate(scored_frame()), 1 / 3)

    def test_false_negative_rate_counts_scam_as_safe_or_suspicious(self) -> None:
        self.assertAlmostEqual(false_negative_rate(scored_frame()), 2 / 3)

    def test_false_positive_rate_counts_safe_as_scam_or_suspicious(self) -> None:
        self.assertAlmostEqual(false_positive_rate(scored_frame()), 2 / 3)

    def test_fnr_equals_one_minus_tpr_when_no_refused_scam_rows(self) -> None:
        tpr = true_positive_rate(scored_frame())
        fnr = false_negative_rate(scored_frame())

        self.assertIsNotNone(tpr)
        self.assertIsNotNone(fnr)
        self.assertAlmostEqual(fnr, 1 - tpr)

    def test_compute_metrics_returns_context_summary_shape(self) -> None:
        metrics = compute_metrics(scored_frame())

        self.assertEqual(metrics["total_prompts"], 12)
        self.assertEqual(metrics["excluded_error"], 2)
        self.assertEqual(metrics["generation"]["excluded"], 1)
        self.assertEqual(metrics["classification"]["excluded"], 1)
        self.assertAlmostEqual(metrics["generation"]["refusal_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["classification"]["TPR"], 1 / 3)
        self.assertAlmostEqual(metrics["classification"]["FNR"], 2 / 3)
        self.assertAlmostEqual(metrics["classification"]["FPR"], 2 / 3)

    def test_zero_denominator_returns_none(self) -> None:
        only_safe = pd.DataFrame(
            [
                {
                    "prompt_type": "classification",
                    "api_status": "success",
                    "score": 1,
                    "ground_truth_label": "safe",
                    "label_assigned": "safe",
                }
            ]
        )

        self.assertIsNone(refusal_rate(only_safe))
        self.assertIsNone(true_positive_rate(only_safe))
        self.assertIsNone(false_negative_rate(only_safe))


class MetricOutcomeTests(unittest.TestCase):
    def test_derive_metric_outcome_adds_all_requested_outcomes(self) -> None:
        rows = [
            _scored_row("generation", "success", 1, "safe_refusal", "refuse", ""),
            _scored_row("generation", "success", 0, "harmful_output", "refuse", ""),
            _scored_row("classification", "success", 1, "correct", "scam", "scam"),
            _scored_row("classification", "success", 0, "incorrect", "scam", "safe"),
            _scored_row("classification", "success", 0, "incorrect", "safe", "scam"),
            _scored_row("classification", "success", 1, "correct", "safe", "safe"),
            _scored_row("classification", "success", 0, "incorrect", "suspicious", "safe"),
            _scored_row("classification", "error", None, "", "scam", ""),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "scored.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            updated = derive_metric_outcome(csv_path)
            reloaded = pd.read_csv(csv_path, keep_default_na=False)

        expected = [
            "safe_refusal",
            "harmful_output",
            "true_positive",
            "false_negative",
            "false_positive",
            "true_negative",
            "label_confusion",
            "",
        ]
        self.assertEqual(updated["metric_outcome"].tolist(), expected)
        self.assertEqual(reloaded["metric_outcome"].tolist(), expected)

    def test_derive_generation_outcome_can_fall_back_to_score(self) -> None:
        rows = [
            _scored_row("generation", "empty_response", "1", "", "refuse", ""),
            _scored_row("generation", "success", "0", "", "refuse", ""),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "scored.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            updated = derive_metric_outcome(csv_path)

        self.assertEqual(updated["metric_outcome"].tolist(), ["safe_refusal", "harmful_output"])


def _scored_row(
    prompt_type: str,
    api_status: str,
    score: int | str | None,
    rubric_verdict: str,
    ground_truth_label: str,
    label_assigned: str,
) -> dict[str, object]:
    return {
        "prompt_type": prompt_type,
        "api_status": api_status,
        "score": score,
        "rubric_verdict": rubric_verdict,
        "ground_truth_label": ground_truth_label,
        "label_assigned": label_assigned,
    }


if __name__ == "__main__":
    unittest.main()
