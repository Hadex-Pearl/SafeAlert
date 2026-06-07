"""Load and validate SafeAlert prompt datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_FIELDS = (
    "id",
    "type",
    "category",
    "category_name",
    "channel",
    "message",
    "label",
    "target_behaviour",
    "notes",
)

REQUIRED_NON_EMPTY_FIELDS = tuple(field for field in DATASET_FIELDS if field != "notes")

VALID_TYPES = {"generation", "classification"}
VALID_CHANNELS = {"SMS", "WhatsApp", "in-app"}
VALID_LABELS = {"refuse", "scam", "suspicious", "safe"}
VALID_TARGET_BEHAVIOURS = {
    "refuse",
    "classify_scam",
    "classify_suspicious",
    "classify_safe",
}

CATEGORY_NAMES = {
    1: "Phishing and fake bank alerts",
    2: "SIM swap fraud",
    3: "Identity theft and credential compromise",
    4: "Fake fintech and bank platform impersonation",
    5: "Investment scams and Ponzi schemes",
    6: "Fake loan app scams",
    7: "Fake job offer scams",
    8: "Government and relief fund impersonation",
}

TARGET_BY_LABEL = {
    "refuse": "refuse",
    "scam": "classify_scam",
    "suspicious": "classify_suspicious",
    "safe": "classify_safe",
}


class PromptValidationError(ValueError):
    """Raised when a SafeAlert prompt dataset fails validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(self._format_message(errors))

    @staticmethod
    def _format_message(errors: list[str]) -> str:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        remaining = len(errors) - 20
        if remaining > 0:
            preview = f"{preview}\n- ... and {remaining} more error(s)"
        return f"Prompt dataset validation failed with {len(errors)} error(s):\n{preview}"


def load_prompts(path: str | Path) -> pd.DataFrame:
    """Load a SafeAlert CSV or JSON prompt dataset and validate it."""
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        prompts = _load_csv(dataset_path)
    elif suffix == ".json":
        prompts = _load_json(dataset_path)
    else:
        raise ValueError(f"Unsupported prompt dataset format: {dataset_path.suffix}")

    return validate_prompts(prompts, source=dataset_path)


def validate_prompts(prompts: pd.DataFrame, source: str | Path = "<memory>") -> pd.DataFrame:
    """Validate and normalize a SafeAlert prompt DataFrame."""
    errors: list[str] = []
    source_label = str(source)
    prompts = prompts.copy()

    _validate_fields(prompts, errors, source_label)
    if errors:
        raise PromptValidationError(errors)

    prompts = prompts.loc[:, DATASET_FIELDS]
    prompts["notes"] = prompts["notes"].fillna("")

    _validate_missing_values(prompts, errors)
    _validate_rows(prompts, errors)

    if errors:
        raise PromptValidationError(errors)

    prompts["category"] = prompts["category"].astype(int)
    return prompts


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read CSV prompt dataset at {path}: {exc}") from exc


def _load_json(path: Path) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8") as dataset_file:
            payload = json.load(dataset_file)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON prompt dataset at {path}: {exc}") from exc

    records = _records_from_json_payload(payload, path)
    if not records:
        return pd.DataFrame(columns=DATASET_FIELDS)

    return pd.DataFrame(records)


def _records_from_json_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
        records = payload["prompts"]
    else:
        raise ValueError(
            f"JSON prompt dataset at {path} must be a list of records or an object with a 'prompts' list."
        )

    invalid_indexes = [index + 1 for index, record in enumerate(records) if not isinstance(record, dict)]
    if invalid_indexes:
        raise ValueError(f"JSON prompt dataset at {path} has non-object records at rows {invalid_indexes}.")

    return records


def _validate_fields(prompts: pd.DataFrame, errors: list[str], source: str) -> None:
    actual_fields = list(prompts.columns)
    missing_fields = [field for field in DATASET_FIELDS if field not in actual_fields]
    extra_fields = [field for field in actual_fields if field not in DATASET_FIELDS]

    if missing_fields:
        errors.append(f"{source}: missing required field(s): {', '.join(missing_fields)}")
    if extra_fields:
        errors.append(f"{source}: unexpected field(s): {', '.join(extra_fields)}")


def _validate_missing_values(prompts: pd.DataFrame, errors: list[str]) -> None:
    for row_index, row in prompts.iterrows():
        row_label = _row_label(row_index, row)
        for field in REQUIRED_NON_EMPTY_FIELDS:
            if _is_missing(row[field]):
                errors.append(f"{row_label}: field '{field}' is required and cannot be blank")


def _validate_rows(prompts: pd.DataFrame, errors: list[str]) -> None:
    for row_index, row in prompts.iterrows():
        row_label = _row_label(row_index, row)
        category = _parse_category(row["category"], row_label, errors)

        _validate_allowed_value(row["type"], VALID_TYPES, "type", row_label, errors)
        _validate_allowed_value(row["channel"], VALID_CHANNELS, "channel", row_label, errors)
        _validate_allowed_value(row["label"], VALID_LABELS, "label", row_label, errors)
        _validate_allowed_value(
            row["target_behaviour"],
            VALID_TARGET_BEHAVIOURS,
            "target_behaviour",
            row_label,
            errors,
        )

        if category is not None:
            expected_name = CATEGORY_NAMES[category]
            if str(row["category_name"]).strip() != expected_name:
                errors.append(
                    f"{row_label}: category_name must be '{expected_name}' for category {category}"
                )

        _validate_type_label_pair(row, row_label, errors)
        _validate_label_target_pair(row, row_label, errors)


def _validate_type_label_pair(row: pd.Series, row_label: str, errors: list[str]) -> None:
    prompt_type = str(row["type"]).strip()
    label = str(row["label"]).strip()

    if prompt_type == "generation" and label != "refuse":
        errors.append(f"{row_label}: generation prompts must use label 'refuse'")
    if prompt_type == "classification" and label == "refuse":
        errors.append(f"{row_label}: classification prompts must use scam, suspicious, or safe labels")


def _validate_label_target_pair(row: pd.Series, row_label: str, errors: list[str]) -> None:
    label = str(row["label"]).strip()
    target_behaviour = str(row["target_behaviour"]).strip()
    expected_target = TARGET_BY_LABEL.get(label)

    if expected_target and target_behaviour != expected_target:
        errors.append(
            f"{row_label}: target_behaviour must be '{expected_target}' when label is '{label}'"
        )


def _parse_category(value: Any, row_label: str, errors: list[str]) -> int | None:
    try:
        category = int(str(value).strip())
    except (TypeError, ValueError):
        errors.append(f"{row_label}: category must be an integer from 1 through 8")
        return None

    if category not in CATEGORY_NAMES:
        errors.append(f"{row_label}: category must be an integer from 1 through 8")
        return None

    return category


def _validate_allowed_value(
    value: Any,
    allowed_values: set[str],
    field: str,
    row_label: str,
    errors: list[str],
) -> None:
    value_text = str(value).strip()
    if value_text not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        errors.append(f"{row_label}: field '{field}' must be one of: {allowed}")


def _is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _row_label(row_index: Any, row: pd.Series) -> str:
    row_number = int(row_index) + 2 if isinstance(row_index, int) else row_index
    prompt_id = row.get("id", "")
    if not _is_missing(prompt_id):
        return f"row {row_number} ({prompt_id})"
    return f"row {row_number}"


def main() -> None:
    """Validate a prompt dataset from the command line."""
    parser = argparse.ArgumentParser(description="Load and validate a SafeAlert prompt dataset.")
    parser.add_argument("path", help="Path to a SafeAlert prompt CSV or JSON file.")
    args = parser.parse_args()

    prompts = load_prompts(args.path)
    print(f"Loaded {len(prompts)} valid prompt(s) from {args.path}.")


if __name__ == "__main__":
    main()
