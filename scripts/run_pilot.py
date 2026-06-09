"""CLI entry point for running SafeAlert pilot evaluations.

Usage
-----
    # Full pilot — all 205 prompts, pre-remediation
    python scripts/run_pilot.py \\
        --dataset dataset/public/safealert_dataset_v1_public.csv \\
        --model gpt4o \\
        --run-type pre_remediation

    # Post-remediation (adds safety system prompt)
    python scripts/run_pilot.py \\
        --dataset dataset/public/safealert_dataset_v1_public.csv \\
        --model llama \\
        --run-type post_remediation

    # Dry-run: validate dataset and show what would run, no API calls
    python scripts/run_pilot.py \\
        --dataset dataset/public/safealert_dataset_v1_public.csv \\
        --model gpt4o \\
        --run-type pre_remediation \\
        --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv

try:
    from prompt_loader import load_prompts
    from gpt4o_mini_api import (
        GPT4oMiniClient,
        FatalAPIError,
        POST_REMEDIATION_SYSTEM_PROMPT as GPT_SYSTEM_PROMPT,
        call_and_log_prompt as gpt_call,
    )
    from llama31_8b_api import (
        Llama31_8BClient,
        FatalAPIError as LlamaFatalAPIError,
        POST_REMEDIATION_SYSTEM_PROMPT as LLAMA_SYSTEM_PROMPT,
        call_and_log_prompt as llama_call,
    )
except ModuleNotFoundError:
    from scripts.prompt_loader import load_prompts
    from scripts.gpt4o_mini_api import (
        GPT4oMiniClient,
        FatalAPIError,
        POST_REMEDIATION_SYSTEM_PROMPT as GPT_SYSTEM_PROMPT,
        call_and_log_prompt as gpt_call,
    )
    from scripts.llama31_8b_api import (
        Llama31_8BClient,
        FatalAPIError as LlamaFatalAPIError,
        POST_REMEDIATION_SYSTEM_PROMPT as LLAMA_SYSTEM_PROMPT,
        call_and_log_prompt as llama_call,
    )


RESULTS_DIR = Path("results") / "raw"

MODEL_CONFIGS = {
    "gpt4o": {
        "display_name": "gpt-4o-mini",
        "client_cls": GPT4oMiniClient,
        "call_fn": gpt_call,
        "fatal_error_cls": FatalAPIError,
        "system_prompt": GPT_SYSTEM_PROMPT,
    },
    "llama": {
        "display_name": "llama-3.1-8b",
        "client_cls": Llama31_8BClient,
        "call_fn": llama_call,
        "fatal_error_cls": LlamaFatalAPIError,
        "system_prompt": LLAMA_SYSTEM_PROMPT,
    },
}


def build_run_id(model_key: str, run_type: str) -> str:
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    model_short = MODEL_CONFIGS[model_key]["display_name"].replace(".", "-")
    run_type_short = "pre" if run_type == "pre_remediation" else "post"
    return f"SA-{model_short}-{run_type_short}-{date_str}"


def run_pilot(
    dataset_path: str,
    model_key: str,
    run_type: str,
    dry_run: bool = False,
) -> int:
    """Run all prompts from dataset_path through the selected model. Returns error count."""
    config = MODEL_CONFIGS[model_key]
    display_name = config["display_name"]

    prompts = load_prompts(dataset_path)
    system_prompt = config["system_prompt"] if run_type == "post_remediation" else None
    run_id = build_run_id(model_key, run_type)
    output_path = RESULTS_DIR / f"{run_id}.jsonl"

    print(f"\nSafeAlert pilot run")
    print(f"  Dataset   : {dataset_path}")
    print(f"  Model     : {display_name}")
    print(f"  Run type  : {run_type}")
    print(f"  Run ID    : {run_id}")
    print(f"  Prompts   : {len(prompts)}")
    print(f"  Output    : {output_path}")
    if system_prompt:
        print(f"  System prompt: yes ({len(system_prompt)} chars)")
    else:
        print(f"  System prompt: none (pre-remediation)")

    if dry_run:
        print(f"\nDry-run mode: dataset validated, no API calls made.")
        return 0

    try:
        client = config["client_cls"]()
    except ValueError as exc:
        print(f"\nError: {exc}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_count = 0
    total = len(prompts)

    print(f"\nStarting run... (writing to {output_path})\n")

    try:
        for i, (_, prompt) in enumerate(prompts.iterrows(), 1):
            print(f"  [{i:03d}/{total}] {prompt['id']}  ", end="", flush=True)
            record = config["call_fn"](
                prompt=prompt,
                client=client,
                output_path=output_path,
                run_id=run_id,
                run_type=run_type,
                system_prompt=system_prompt,
            )
            status = record["api_status"]
            tokens = record.get("response_tokens", "n/a")
            latency = record.get("api_latency_ms", "n/a")
            print(f"status={status}  tokens={tokens}  {latency}ms")
            if status == "error":
                error_count += 1

    except (FatalAPIError, LlamaFatalAPIError) as exc:
        print(f"\nFatal API error — run stopped: {exc}")
        print(f"Records written before error: see {output_path}")
        return 1

    print(f"\nRun complete.")
    print(f"  Total   : {total}")
    print(f"  Errors  : {error_count}")
    print(f"  Output  : {output_path}")
    print(f"\nNext step: open notebooks/safealert_scorer.ipynb to score responses.")
    return error_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SafeAlert pilot evaluation against one model."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to a SafeAlert prompt CSV or JSON file.",
    )
    parser.add_argument(
        "--model", required=True, choices=("gpt4o", "llama"),
        help="Which model to run.",
    )
    parser.add_argument(
        "--run-type", required=True,
        choices=("pre_remediation", "post_remediation"),
        help="pre_remediation = no system prompt. post_remediation = safety system prompt applied.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset and print run details without making any API calls.",
    )
    args = parser.parse_args()

    load_dotenv()
    error_count = run_pilot(
        dataset_path=args.dataset,
        model_key=args.model,
        run_type=args.run_type,
        dry_run=args.dry_run,
    )
    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
