"""End-to-end test: run 10 prompts per model and verify outputs are readable.

Usage
-----
    # Classification prompts only (public dataset)
    python scripts/test_e2e.py \\
        --dataset dataset/public/safealert_dataset_v1_public.csv

    # Both generation and classification (requires private dataset path)
    python scripts/test_e2e.py \\
        --dataset dataset/public/safealert_dataset_v1_public.csv \\
        --private-dataset path/to/safealert_dataset_v1_private.csv

    # Test one model only
    python scripts/test_e2e.py \\
        --dataset dataset/public/safealert_dataset_v1_public.csv \\
        --model gpt4o

Exit codes
----------
    0  All prompts returned a response and JSONL output passed validation.
    1  One or more prompts errored or output validation failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from prompt_loader import load_prompts
    from response_recorder import validate_raw_response_record
    from gpt4o_mini_api import GPT4oMiniClient, call_and_log_prompt as gpt_call, FatalAPIError
    from llama31_8b_api import Llama31_8BClient, call_and_log_prompt as llama_call
    from llama31_8b_api import FatalAPIError as LlamaFatalAPIError
except ModuleNotFoundError:
    from scripts.prompt_loader import load_prompts
    from scripts.response_recorder import validate_raw_response_record
    from scripts.gpt4o_mini_api import GPT4oMiniClient, call_and_log_prompt as gpt_call, FatalAPIError
    from scripts.llama31_8b_api import Llama31_8BClient, call_and_log_prompt as llama_call
    from scripts.llama31_8b_api import FatalAPIError as LlamaFatalAPIError


N_PROMPTS = 10
RESULTS_DIR = Path("results") / "raw"
SEPARATOR = "─" * 70


def sample_prompts(public_path: str, private_path: str | None = None, n: int = N_PROMPTS) -> list:
    """Sample n prompts, mixing generation and classification if private path provided."""
    public = load_prompts(public_path)
    frames = [public]

    if private_path:
        private = load_prompts(private_path)
        frames.append(private)

    all_prompts = pd.concat(frames, ignore_index=True)

    generation = all_prompts[all_prompts["type"] == "generation"]
    classification = all_prompts[all_prompts["type"] == "classification"]

    samples = []

    if not generation.empty:
        # One generation prompt per category, up to 5
        gen_sample = (
            generation.sort_values("category")
            .groupby("category")
            .first()
            .reset_index(drop=True)
            .head(5)
        )
        samples.extend(gen_sample.to_dict("records"))

    if not classification.empty:
        # Spread across labels — roughly equal scam/suspicious/safe
        remaining = n - len(samples)
        per_label = max(1, remaining // 3)
        for label in ("scam", "suspicious", "safe"):
            subset = classification[classification["label"] == label]
            if not subset.empty:
                samples.extend(
                    subset.sample(min(per_label, len(subset)), random_state=42).to_dict("records")
                )

    return samples[:n]


def run_model(
    model_name: str,
    client,
    call_fn,
    fatal_error_cls,
    prompts: list,
    run_type: str = "pre_remediation",
) -> tuple[list[dict], list[str], Path]:
    """Run all prompts through one model. Returns (records, api_errors, output_path)."""
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    model_short = model_name.replace(".", "-")
    run_type_short = "pre" if run_type == "pre_remediation" else "post"
    run_id = f"SA-{model_short}-{run_type_short}-{date_str}-e2e"
    output_path = RESULTS_DIR / f"{run_id}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records, api_errors = [], []

    print(f"\n{SEPARATOR}")
    print(f"Model  : {model_name}")
    print(f"Run ID : {run_id}")
    print(f"Output : {output_path}")
    print(SEPARATOR)

    for i, prompt in enumerate(prompts, 1):
        prompt_id = prompt["id"]
        prompt_type = prompt["type"]
        label = prompt["label"]
        print(f"\n[{i:02d}/{len(prompts)}] {prompt_id}  type={prompt_type}  label={label}")
        print(f"        {textwrap.shorten(str(prompt['message']), 80, placeholder='...')}")

        try:
            record = call_fn(
                prompt=prompt,
                client=client,
                output_path=output_path,
                run_id=run_id,
                run_type=run_type,
            )
        except fatal_error_cls as exc:
            print(f"\n        FATAL ERROR — stopping run: {exc}")
            break

        status = record["api_status"]
        tokens = record.get("response_tokens", "n/a")
        latency = record.get("api_latency_ms", "n/a")
        print(f"        status={status}  tokens={tokens}  latency={latency}ms")

        if status == "success":
            preview = textwrap.shorten(record["raw_response"], 120, placeholder="...")
            print(f"        response: {preview}")
        elif status == "error":
            err = record.get("reviewer_notes", "unknown error")
            print(f"        ERROR: {err}")
            api_errors.append(f"{prompt_id}: {err}")
        elif status == "empty_response":
            print("        WARNING: empty response received")

        records.append(record)

    return records, api_errors, output_path


def validate_jsonl(path: Path) -> list[str]:
    """Validate every record in a JSONL file. Returns list of validation errors."""
    if not path.exists():
        return [f"{path} does not exist"]
    errors = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                validate_raw_response_record(json.loads(line))
            except Exception as exc:
                errors.append(f"line {line_num}: {exc}")
    return errors


def print_summary(model_name: str, records: list, api_errors: list, jsonl_errors: list) -> bool:
    """Print a human-readable test summary. Returns True if all passed."""
    total = len(records)
    successes = sum(1 for r in records if r["api_status"] == "success")
    empties = sum(1 for r in records if r["api_status"] == "empty_response")
    errors = sum(1 for r in records if r["api_status"] == "error")

    print(f"\n{'═' * 70}")
    print(f"  SUMMARY — {model_name}")
    print(f"{'═' * 70}")
    print(f"  Total prompts   : {total}")
    print(f"  Successful      : {successes}")
    print(f"  Empty response  : {empties}")
    print(f"  API errors      : {errors}")
    print(f"  JSONL errors    : {len(jsonl_errors)}")

    if api_errors:
        print("\n  API error details:")
        for e in api_errors:
            print(f"    ✗ {e}")
    if jsonl_errors:
        print("\n  JSONL validation errors:")
        for e in jsonl_errors:
            print(f"    ✗ {e}")

    passed = errors == 0 and len(jsonl_errors) == 0
    print(f"\n  Result          : {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'═' * 70}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SafeAlert end-to-end test: run 10 prompts per model."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to the public SafeAlert classification dataset (CSV or JSON).",
    )
    parser.add_argument(
        "--private-dataset", default=None,
        help="Optional path to the private generation dataset. Adds generation prompts to the test.",
    )
    parser.add_argument(
        "--model", choices=("gpt4o", "llama", "both"), default="both",
        help="Which model(s) to test (default: both).",
    )
    parser.add_argument(
        "--run-type",
        choices=("pre_remediation", "post_remediation"),
        default="pre_remediation",
    )
    args = parser.parse_args()

    load_dotenv()

    print(f"\nSafeAlert end-to-end test")
    print(f"Dataset       : {args.dataset}")
    if args.private_dataset:
        print(f"Private dataset: {args.private_dataset}")
    print(f"Model(s)      : {args.model}")
    print(f"Run type      : {args.run_type}")

    prompts = sample_prompts(args.dataset, args.private_dataset)

    print(f"\nSampled {len(prompts)} prompts:")
    for p in prompts:
        print(f"  {p['id']:30s}  type={p['type']:14s}  cat={p['category']}  label={p['label']}")

    all_passed = True

    model_configs = {
        "gpt4o": ("gpt-4o-mini", GPT4oMiniClient, gpt_call, FatalAPIError),
        "llama": ("llama-3.1-8b", Llama31_8BClient, llama_call, LlamaFatalAPIError),
    }

    models_to_run = (
        ["gpt4o", "llama"] if args.model == "both" else [args.model]
    )

    for model_key in models_to_run:
        display_name, client_cls, call_fn, fatal_cls = model_configs[model_key]
        try:
            client = client_cls()
        except ValueError as exc:
            print(f"\n✗ {display_name}: {exc}")
            all_passed = False
            continue

        records, api_errors, output_path = run_model(
            display_name, client, call_fn, fatal_cls, prompts, args.run_type
        )
        jsonl_errors = validate_jsonl(output_path)
        passed = print_summary(display_name, records, api_errors, jsonl_errors)
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
