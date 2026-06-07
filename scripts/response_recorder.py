"""Record SafeAlert raw model responses as JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


RAW_RESPONSE_FIELDS = (
    "run_id",
    "prompt_id",
    "prompt_type",
    "category",
    "category_name",
    "ground_truth_label",
    "target_behaviour",
    "model",
    "run_type",
    "timestamp_utc",
    "prompt_text",
    "raw_response",
    "response_tokens",
    "api_latency_ms",
    "api_status",
    "rubric_verdict",
    "score",
    "label_assigned",
    "error_type",
    "reviewer_notes",
)


@dataclass(frozen=True)
class RawResponsePayload:
    """Model response metadata needed to write one raw SafeAlert record."""

    raw_response: str
    response_tokens: int | None
    api_latency_ms: int
    api_status: str
    reviewer_notes: str = ""


def build_raw_response_record(
    prompt: pd.Series | dict[str, Any],
    payload: RawResponsePayload,
    run_id: str,
    model: str,
    run_type: str,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build one raw response record using the schema from context.md."""
    prompt_data = prompt.to_dict() if isinstance(prompt, pd.Series) else dict(prompt)
    record = {
        "run_id": run_id,
        "prompt_id": prompt_data["id"],
        "prompt_type": prompt_data["type"],
        "category": int(prompt_data["category"]),
        "category_name": prompt_data["category_name"],
        "ground_truth_label": prompt_data["label"],
        "target_behaviour": prompt_data["target_behaviour"],
        "model": model,
        "run_type": run_type,
        "timestamp_utc": timestamp_utc or utc_timestamp(),
        "prompt_text": prompt_data["message"],
        "raw_response": payload.raw_response,
        "response_tokens": payload.response_tokens,
        "api_latency_ms": payload.api_latency_ms,
        "api_status": payload.api_status,
        "rubric_verdict": "",
        "score": None,
        "label_assigned": "",
        "error_type": "",
        "reviewer_notes": payload.reviewer_notes,
    }
    validate_raw_response_record(record)
    return record


def append_response_record(output_path: str | Path, record: dict[str, Any]) -> None:
    """Append one validated raw response record to a JSONL file."""
    validate_raw_response_record(record)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_raw_response(
    output_path: str | Path,
    prompt: pd.Series | dict[str, Any],
    payload: RawResponsePayload,
    run_id: str,
    model: str,
    run_type: str,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build, validate, and append one raw response record."""
    record = build_raw_response_record(
        prompt=prompt,
        payload=payload,
        run_id=run_id,
        model=model,
        run_type=run_type,
        timestamp_utc=timestamp_utc,
    )
    append_response_record(output_path, record)
    return record


def validate_raw_response_record(record: dict[str, Any]) -> None:
    """Ensure a raw response record has exactly the expected fields."""
    actual_fields = tuple(record.keys())
    if actual_fields != RAW_RESPONSE_FIELDS:
        missing = [field for field in RAW_RESPONSE_FIELDS if field not in record]
        extra = [field for field in actual_fields if field not in RAW_RESPONSE_FIELDS]
        raise ValueError(
            "Raw response record field mismatch. "
            f"Missing: {missing or 'none'}. Extra: {extra or 'none'}."
        )


def utc_timestamp() -> str:
    """Return a SafeAlert UTC timestamp string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    """Validate an existing raw JSONL response file."""
    parser = argparse.ArgumentParser(description="Validate SafeAlert raw response JSONL records.")
    parser.add_argument("path", help="Path to a raw JSONL response file.")
    args = parser.parse_args()

    path = Path(args.path)
    count = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                validate_raw_response_record(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid raw response record: {exc}") from exc
            count += 1

    print(f"Validated {count} raw response record(s) from {path}.")


if __name__ == "__main__":
    main()
