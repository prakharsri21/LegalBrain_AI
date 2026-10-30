"""Create a reproducible human-review pilot from language-validation JSONL.

This script selects review rows only. It assigns no language truth labels and
does not modify the 10,000-record source file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE = Path("data/reports/language-validation-v1/language_review.jsonl")
DEFAULT_TEMPLATE = Path(
    "data/reports/language-validation-v1/language_review_template.json"
)
DEFAULT_OUTPUT = Path("data/reports/language-validation-pilot-v1")
DEFAULT_SEED = 20261030
LABEL_FIELDS = (
    "human_question_language",
    "human_response_language",
    "human_mixed_language_status",
    "question_genuinely_english",
    "response_genuinely_english",
    "reviewer_id",
    "review_notes",
)

PILOT_CSV_COLUMNS = (
    "source_row_id",
    "source_shard",
    "source_row_number_1_based",
    "dataset_revision",
    "question",
    "response",
    "context_role",
    "context_sha256",
    "field_lengths_chars_context",
    "field_lengths_chars_question",
    "field_lengths_chars_response",
    "candidate_strata",
    "sampling_basis",
    "uniform_core_inclusion_probability",
    "supplemental_selection_strata",
    "audit_flags",
    "question_prediction_status",
    "question_prediction_language",
    "question_prediction_classifier_score",
    "question_prediction_score_is_probability",
    "question_prediction_input_characters",
    "question_prediction_classified_characters",
    "question_prediction_input_truncated",
    "question_prediction_error_type",
    "response_prediction_status",
    "response_prediction_language",
    "response_prediction_classifier_score",
    "response_prediction_score_is_probability",
    "response_prediction_input_characters",
    "response_prediction_classified_characters",
    "response_prediction_input_truncated",
    "response_prediction_error_type",
    "pilot_selection_type",
    "pilot_selection_group",
    "pilot_selection_strata",
    "pilot_seed",
    "pilot_conditional_inclusion_probability",
    "pilot_selection_method",
    *LABEL_FIELDS,
)

LANGUAGE_STRATA_PREFIXES = (
    "predicted_question_language:",
    "predicted_response_language:",
)
MIXED_STRATA = ("mixed_language_looking", "question_response_prediction_disagreement")
SHORT_STRATA = ("short_text_under_40_chars",)
POTENTIAL_ERROR_STRATA = (
    "audit_control_or_replacement_flag",
    "legal_citation",
    "name_like_capitalized_sequence",
    "quotation_mark",
    "long_text_at_least_1000_chars",
    "replacement_character",
    "suspicious_control_character",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            records.append(record)
    return records


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _source_type(record: dict[str, Any]) -> str:
    basis = record.get("sampling_basis")
    if not isinstance(basis, list) or len(basis) != 1:
        raise ValueError(
            f"Expected exactly one original sampling_basis for {record.get('source_row_id')}"
        )
    if basis[0] == "uniform_core":
        return "probability_sample"
    if basis[0] == "supplemental_targeted":
        return "targeted_supplemental"
    raise ValueError(
        f"Unsupported original sampling_basis {basis!r} for {record.get('source_row_id')}"
    )


def _language_count_for_stratum(
    stratum: str, prediction_counts: dict[str, Any]
) -> int:
    if stratum.startswith("predicted_question_language:"):
        field = "question"
        language = stratum.split(":", 1)[1]
    elif stratum.startswith("predicted_response_language:"):
        field = "response"
        language = stratum.split(":", 1)[1]
    else:
        return 0
    return int(prediction_counts.get(field, {}).get(language, 0))


def _round_robin_select(
    candidates: list[dict[str, Any]],
    strata: list[str],
    count: int,
    seed: int,
    selected_ids: set[str],
) -> list[tuple[dict[str, Any], str]]:
    """Select unique rows across strata, shuffling each stratum reproducibly."""
    if count <= 0 or not strata:
        return []
    queues: dict[str, list[dict[str, Any]]] = {}
    positions: Counter[str] = Counter()
    for stratum in strata:
        matching = [
            record
            for record in candidates
            if stratum in record.get("candidate_strata", [])
        ]
        matching.sort(key=lambda record: str(record.get("source_row_id", "")))
        random.Random(_stable_seed(seed, f"target:{stratum}")).shuffle(matching)
        queues[stratum] = matching

    chosen: list[tuple[dict[str, Any], str]] = []
    while len(chosen) < count:
        progressed = False
        for stratum in strata:
            queue = queues[stratum]
            while positions[stratum] < len(queue):
                record = queue[positions[stratum]]
                positions[stratum] += 1
                source_row_id = record.get("source_row_id")
                if source_row_id in selected_ids:
                    continue
                selected_ids.add(source_row_id)
                chosen.append((record, stratum))
                progressed = True
                break
            if len(chosen) >= count:
                break
        if not progressed:
            break
    return chosen


def _mark_selection(
    record: dict[str, Any],
    selection_type: str,
    selection_group: str,
    selection_strata: list[str],
    seed: int,
    conditional_probability: float | None,
    method: str,
) -> dict[str, Any]:
    # Context is intentionally omitted from pilot review artifacts. The row ID,
    # shard/row location, context role, and original digest retain provenance.
    result = {key: value for key, value in record.items() if key != "context"}
    result["pilot_selection"] = {
        "pilot_selection_type": selection_type,
        "pilot_selection_group": selection_group,
        "pilot_selection_strata": selection_strata,
        "pilot_seed": seed,
        "pilot_conditional_inclusion_probability": conditional_probability,
        "pilot_selection_method": method,
    }
    return result


def create_pilot_records(
    records: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    probability_count: int = 400,
    targeted_count: int = 100,
    prediction_counts: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if probability_count < 0 or targeted_count < 0:
        raise ValueError("Pilot sample counts cannot be negative")
    prediction_counts = prediction_counts or {}

    source_ids = [record.get("source_row_id") for record in records]
    if any(not isinstance(source_id, str) or not source_id for source_id in source_ids):
        raise ValueError("Every source record must have a non-empty source_row_id")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Source review file contains duplicate source_row_id values")

    probability_records: list[dict[str, Any]] = []
    targeted_records: list[dict[str, Any]] = []
    for record in records:
        source_type = _source_type(record)
        if source_type == "probability_sample":
            probability_records.append(record)
        else:
            targeted_records.append(record)
    if probability_count > len(probability_records):
        raise ValueError(
            f"Requested {probability_count} probability rows, but only "
            f"{len(probability_records)} are available"
        )
    if targeted_count > len(targeted_records):
        raise ValueError(
            f"Requested {targeted_count} targeted rows, but only "
            f"{len(targeted_records)} are available"
        )

    probability_records.sort(key=lambda record: record["source_row_id"])
    targeted_records.sort(key=lambda record: record["source_row_id"])
    core_rng = random.Random(_stable_seed(seed, "probability-core"))
    selected_probability = core_rng.sample(probability_records, probability_count)
    selected_ids = {record["source_row_id"] for record in selected_probability}

    target_quota = {
        "uncommon_predicted_language": round(targeted_count * 0.40),
        "mixed_language_indicator": round(targeted_count * 0.25),
        "short_field": round(targeted_count * 0.15),
    }
    target_quota["potential_classifier_error"] = (
        targeted_count - sum(target_quota.values())
    )
    selected_targeted: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    target_strata_counts: Counter[str] = Counter()

    def add_group(
        group: str,
        selected: list[tuple[dict[str, Any], str]],
        selection_type: str = "targeted_supplemental",
    ) -> None:
        for record, stratum in selected:
            source_id = record["source_row_id"]
            if source_id in selected_ids:
                continue
            selected_ids.add(source_id)
            strata = [group, stratum]
            selected_targeted.append(
                _mark_selection(
                    record,
                    selection_type,
                    group,
                    strata,
                    seed,
                    None,
                    "Seeded random selection within the original targeted candidate stratum; no single inclusion probability applies.",
                )
            )
            group_counts[group] += 1
            target_strata_counts[stratum] += 1

    language_strata = sorted(
        {
            stratum
            for record in targeted_records
            for stratum in record.get("candidate_strata", [])
            if stratum.startswith(LANGUAGE_STRATA_PREFIXES)
            and not stratum.endswith(":en")
        },
        key=lambda stratum: (
            _language_count_for_stratum(stratum, prediction_counts),
            stratum,
        ),
    )
    uncommon = _round_robin_select(
        targeted_records,
        language_strata,
        target_quota["uncommon_predicted_language"],
        seed,
        selected_ids,
    )
    # The helper marks IDs as selected; add_group uses the same IDs, so release
    # them first and let add_group attach the pilot selection metadata.
    selected_ids.difference_update(record["source_row_id"] for record, _ in uncommon)
    add_group("uncommon_predicted_language", uncommon)

    mixed = _round_robin_select(
        targeted_records,
        list(MIXED_STRATA),
        target_quota["mixed_language_indicator"],
        seed,
        selected_ids,
    )
    selected_ids.difference_update(record["source_row_id"] for record, _ in mixed)
    add_group("mixed_language_indicator", mixed)

    short = _round_robin_select(
        targeted_records,
        list(SHORT_STRATA),
        target_quota["short_field"],
        seed,
        selected_ids,
    )
    selected_ids.difference_update(record["source_row_id"] for record, _ in short)
    add_group("short_field", short)

    potential_errors = _round_robin_select(
        targeted_records,
        list(POTENTIAL_ERROR_STRATA),
        target_quota["potential_classifier_error"],
        seed,
        selected_ids,
    )
    selected_ids.difference_update(
        record["source_row_id"] for record, _ in potential_errors
    )
    add_group("potential_classifier_error", potential_errors)

    remaining_targeted_slots = targeted_count - len(selected_targeted)
    if remaining_targeted_slots:
        fallback_rng = random.Random(_stable_seed(seed, "targeted-fallback"))
        fallback_candidates = [
            record
            for record in targeted_records
            if record["source_row_id"] not in selected_ids
        ]
        fallback_candidates.sort(key=lambda record: record["source_row_id"])
        fallback_rng.shuffle(fallback_candidates)
        for record in fallback_candidates[:remaining_targeted_slots]:
            selected_ids.add(record["source_row_id"])
            strata = list(record.get("candidate_strata", []))
            selected_targeted.append(
                _mark_selection(
                    record,
                    "targeted_supplemental",
                    "targeted_fallback",
                    ["targeted_fallback", *strata],
                    seed,
                    None,
                    "Seeded random selection from remaining original targeted supplemental records.",
                )
            )
            group_counts["targeted_fallback"] += 1

    if len(selected_targeted) != targeted_count:
        raise RuntimeError(
            f"Could select only {len(selected_targeted)} of {targeted_count} targeted pilot rows"
        )

    probability_probability = (
        probability_count / len(probability_records) if probability_records else None
    )
    selected_probability_pilot = [
        _mark_selection(
            record,
            "probability_sample",
            "probability_core",
            ["uniform_core_random_sample"],
            seed,
            probability_probability,
            "Fixed-seed simple random sample without replacement from the original uniform_core subset.",
        )
        for record in selected_probability
    ]

    pilot = selected_probability_pilot + selected_targeted
    pilot.sort(
        key=lambda record: (
            record["pilot_selection"]["pilot_selection_type"],
            record.get("source_shard", ""),
            record.get("source_row_number_1_based", 0),
        )
    )
    if len({record["source_row_id"] for record in pilot}) != len(pilot):
        raise RuntimeError("Pilot selection contains duplicate source_row_id values")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_seed": seed,
        "source_record_count": len(records),
        "source_probability_sample_count": len(probability_records),
        "source_targeted_supplemental_count": len(targeted_records),
        "pilot_record_count": len(pilot),
        "pilot_probability_sample_count": len(selected_probability_pilot),
        "pilot_targeted_supplemental_count": len(selected_targeted),
        "pilot_targeted_group_counts": dict(group_counts),
        "pilot_targeted_selected_strata_counts": dict(target_strata_counts),
        "targeted_quota_plan": target_quota,
        "uncommon_language_selection_order": "Non-English predicted field-language strata ordered by frequency in the existing 10,000-record sample summary, rarest first. These counts describe that sample, not the corpus.",
        "targeted_selection_note": "Candidate strata are indicators for human review, not confirmed classifier errors or language truth. Targeted selections have no single inclusion probability.",
        "pilot_source_context_included": False,
        "provenance_retained": [
            "source_row_id",
            "source_shard",
            "source_row_number_1_based",
            "context_role",
            "context_sha256",
        ],
        "label_schema_source": "language_review_template.json from the source review directory; label field names and allowed values are copied unchanged.",
        "raw_data_modified": False,
        "source_review_file_modified": False,
    }
    return pilot, summary


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in (
        "source_row_id",
        "source_shard",
        "source_row_number_1_based",
        "dataset_revision",
        "question",
        "response",
        "context_role",
        "context_sha256",
        "uniform_core_inclusion_probability",
        *LABEL_FIELDS,
    ):
        row[key] = record.get(key, "")
    lengths = record.get("field_lengths_chars", {})
    for field in ("context", "question", "response"):
        row[f"field_lengths_chars_{field}"] = lengths.get(field, "")
    for key in (
        "candidate_strata",
        "sampling_basis",
        "supplemental_selection_strata",
        "audit_flags",
    ):
        row[key] = _json_cell(record.get(key, []))
    for field in ("question", "response"):
        prediction = record.get(f"{field}_prediction", {})
        for pred_key in (
            "status",
            "language",
            "classifier_score",
            "score_is_probability",
            "input_characters",
            "classified_characters",
            "input_truncated",
            "error_type",
        ):
            row[f"{field}_prediction_{pred_key}"] = prediction.get(pred_key, "")
    pilot = record.get("pilot_selection", {})
    row.update(
        {
            "pilot_selection_type": pilot.get("pilot_selection_type", ""),
            "pilot_selection_group": pilot.get("pilot_selection_group", ""),
            "pilot_selection_strata": _json_cell(
                pilot.get("pilot_selection_strata", [])
            ),
            "pilot_seed": pilot.get("pilot_seed", ""),
            "pilot_conditional_inclusion_probability": pilot.get(
                "pilot_conditional_inclusion_probability", ""
            ),
            "pilot_selection_method": pilot.get("pilot_selection_method", ""),
        }
    )
    return row


def render_pilot_instructions(
    template: dict[str, Any], seed: int, output_dir: Path = DEFAULT_OUTPUT
) -> str:
    field_lines = []
    for name, details in template["fields"].items():
        allowed = details.get("allowed_values", "free text")
        field_lines.append(f"- `{name}`: {allowed}")
    allowed_lines = "\n".join(field_lines)
    template_instructions = "\n".join(
        f"- {instruction}" for instruction in template.get("instructions", [])
    )
    output_location = output_dir
    if output_location.is_absolute():
        try:
            output_location = output_location.relative_to(PROJECT_ROOT)
        except ValueError:
            output_location = output_dir
    output_prefix = output_location.as_posix().rstrip("/")
    return f"""# Language validation human-review pilot

This pilot contains 500 unique pairs selected from the existing 10,000-record review file with fixed seed `{seed}`: 400 records sampled randomly from the `uniform_core` probability subset and 100 targeted records sampled within existing targeted strata. The targeted group prioritizes uncommon predicted labels, mixed-language indicators, short fields, and text/prediction patterns that may expose classifier errors. Those patterns are review candidates, not confirmed errors.

## How to label

Label `question` and `response` independently, using only the corresponding field. Do not use the source `context` to decide either field's language or Englishness. The pilot omits the full context; it retains source row/shard provenance and a context digest.

The language label fields are `human_question_language` and `human_response_language`. Use an ISO language code, `mixed`, or `uncertain`. Fill `human_mixed_language_status` and the two Englishness fields using the existing template values below. Add a reviewer identifier and notes where useful.

{template_instructions}

## Existing label fields and allowed values

{allowed_lines}

## Save and evaluate

1. Use `language_review_pilot.csv` as the spreadsheet-friendly file. Do not change row order, `source_row_id`, predictions, scores, `sampling_basis`, shard provenance, or pilot-selection metadata. Edit only the seven human-label/reviewer columns.
2. Save the completed spreadsheet as `language_review_pilot_completed.csv` in this directory; keep the original pilot CSV unchanged.
3. From the repository root, convert the completed CSV into evaluator-compatible JSONL:

   ```bash
   uv run python scripts/data_pipeline/audit/complete_language_review_pilot.py \\
     {output_prefix}/language_review_pilot.jsonl \\
     {output_prefix}/language_review_pilot_completed.csv \\
     {output_prefix}/language_review_pilot_completed.jsonl \\
     --template {output_prefix}/language_review_template.json
   ```

4. Run the existing evaluator:

   ```bash
   uv run python scripts/data_pipeline/audit/evaluate_language_review.py \\
     {output_prefix}/language_review_pilot_completed.jsonl
   ```

The evaluator writes summary metrics and error cases next to the completed JSONL. Inspect probability-sample and targeted-supplemental metrics separately. Do not convert classifier scores to percentages or treat targeted-sample counts as corpus estimates. Human review does not itself authorize an automatic language filter.
"""


def ensure_fresh_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")


def write_pilot_outputs(
    output_dir: Path,
    pilot: list[dict[str, Any]],
    summary: dict[str, Any],
    template: dict[str, Any],
    seed: int,
) -> None:
    ensure_fresh_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "language_review_pilot.jsonl"
    csv_path = output_dir / "language_review_pilot.csv"
    instructions_path = output_dir / "language_review_pilot_instructions.md"
    template_path = output_dir / "language_review_template.json"
    summary_path = output_dir / "language_review_pilot_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in pilot:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in pilot)
    template_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    instructions_path.write_text(
        render_pilot_instructions(template, seed, output_dir), encoding="utf-8"
    )
    summary["output_files"] = {
        "review_csv": csv_path.name,
        "evaluation_jsonl": jsonl_path.name,
        "instructions": instructions_path.name,
        "label_template": template_path.name,
        "pilot_summary": summary_path.name,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a seeded human-review pilot from a language-review JSONL."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--probability-count", type=int, default=400)
    parser.add_argument("--targeted-count", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_fresh_output_dir(args.output_dir)
    records = read_jsonl(args.source)
    summary_path = args.summary or args.source.with_name(
        "language_classification_summary.json"
    )
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    template = json.loads(args.template.read_text(encoding="utf-8"))
    if set(LABEL_FIELDS) - set(template.get("fields", {})):
        raise ValueError("The source template is missing existing review fields")

    pilot, summary = create_pilot_records(
        records,
        seed=args.seed,
        probability_count=args.probability_count,
        targeted_count=args.targeted_count,
        prediction_counts=source_summary.get(
            "sample_prediction_counts_only_not_corpus_estimates", {}
        ),
    )
    summary.update(
        {
            "source_review_file": str(args.source),
            "source_summary_file": str(summary_path),
            "source_template_file": str(args.template),
            "source_summary_warning": "Source prediction frequencies describe only the existing 10,000-record sample; they are used to order targeted predicted-label strata, not as corpus estimates.",
        }
    )
    write_pilot_outputs(args.output_dir, pilot, summary, template, args.seed)
    print(
        f"Wrote {len(pilot)} unique review pairs to {args.output_dir}", flush=True
    )


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI run
    main()
