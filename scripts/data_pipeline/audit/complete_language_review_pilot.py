"""Merge completed pilot CSV labels into evaluation-compatible JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_pipeline.audit.create_language_review_pilot import (
    DEFAULT_TEMPLATE,
    LABEL_FIELDS,
    read_jsonl,
)

LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")


def _validate_human_labels(
    values: dict[str, str], template: dict[str, Any], source_row_id: str
) -> dict[str, str]:
    output = {key: value.strip() for key, value in values.items()}
    for field in (
        "human_question_language",
        "human_response_language",
    ):
        value = output[field].lower()
        if value in {"mixed", "uncertain"} or LANGUAGE_CODE.fullmatch(value):
            output[field] = value
        else:
            raise ValueError(
                f"{source_row_id}: {field} must be an ISO 639-1/639-3 code, mixed, or uncertain"
            )

    for field in (
        "human_mixed_language_status",
        "question_genuinely_english",
        "response_genuinely_english",
    ):
        allowed = template["fields"][field]["allowed_values"]
        value = output[field].lower()
        if value not in allowed:
            raise ValueError(f"{source_row_id}: invalid {field} value {value!r}")
        output[field] = value

    if not output["reviewer_id"]:
        raise ValueError(f"{source_row_id}: reviewer_id is required")
    return output


def merge_completed_csv(
    pilot_jsonl: Path,
    completed_csv: Path,
    output_jsonl: Path,
    template_path: Path = DEFAULT_TEMPLATE,
) -> int:
    if output_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite completed JSONL: {output_jsonl}")
    records = read_jsonl(pilot_jsonl)
    template = json.loads(template_path.read_text(encoding="utf-8"))

    labels_by_id: dict[str, dict[str, str]] = {}
    with completed_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"source_row_id", *LABEL_FIELDS}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Completed CSV is missing columns: {', '.join(sorted(missing_columns))}"
            )
        for row_number, row in enumerate(reader, start=2):
            source_row_id = (row.get("source_row_id") or "").strip()
            if not source_row_id:
                raise ValueError(f"Completed CSV line {row_number} has no source_row_id")
            if source_row_id in labels_by_id:
                raise ValueError(f"Duplicate source_row_id in completed CSV: {source_row_id}")
            values = {field: row.get(field, "") or "" for field in LABEL_FIELDS}
            labels_by_id[source_row_id] = _validate_human_labels(
                values, template, source_row_id
            )

    record_ids = [record.get("source_row_id") for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Pilot JSONL contains duplicate source_row_id values")
    if set(record_ids) != set(labels_by_id):
        missing = set(record_ids) - set(labels_by_id)
        extra = set(labels_by_id) - set(record_ids)
        raise ValueError(
            f"CSV/JSONL source_row_id mismatch (missing labels={len(missing)}, extra={len(extra)})"
        )

    for record in records:
        record.update(labels_by_id[record["source_row_id"]])
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge reviewed CSV labels into the pilot's JSONL without changing metadata."
    )
    parser.add_argument("pilot_jsonl", type=Path)
    parser.add_argument("completed_csv", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = merge_completed_csv(
        args.pilot_jsonl, args.completed_csv, args.output_jsonl, args.template
    )
    print(f"Wrote {count} labeled rows to {args.output_jsonl}", flush=True)


if __name__ == "__main__":  # pragma: no cover - covered by unit tests
    main()
