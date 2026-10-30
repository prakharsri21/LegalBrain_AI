"""Bounded-memory Stage 3 cleaning and outcome pass for raw Parquet shards.

This runner applies the approved label-based English/Hindi gate, deterministic
review rules, and global exact-duplicate handling. It writes derived Parquet
partitions and a provenance ledger without modifying raw source files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import resource
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq
import pyarrow as pa

try:  # Support both ``python -m`` and the documented repo-root script command.
    from scripts.data_pipeline.audit.validate_language import (
        classify_field,
        language_classifier_version,
    )
    from scripts.data_pipeline.audit.run_audit import (
        CONTROL_CHARACTERS,
        FORM_FEED,
        MISSING_LEGAL_REFERENCE_NUMBER,
        get_record_flags,
        finalize_field_stats,
        new_field_stats,
        observe_field,
        stable_hash,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script path fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.data_pipeline.audit.validate_language import (
        classify_field,
        language_classifier_version,
    )
    from scripts.data_pipeline.audit.run_audit import (
        CONTROL_CHARACTERS,
        FORM_FEED,
        MISSING_LEGAL_REFERENCE_NUMBER,
        get_record_flags,
        finalize_field_stats,
        new_field_stats,
        observe_field,
        stable_hash,
    )

try:
    from .language_policy import (
        LanguagePolicy,
        build_language_outcome_record,
        decide_language_outcome,
        normalize_language_code,
    )
except ImportError:  # pragma: no cover - direct script path fallback
    from language_policy import (  # type: ignore[no-redef]
        LanguagePolicy,
        build_language_outcome_record,
        decide_language_outcome,
        normalize_language_code,
    )

REQUIRED_COLUMNS = ("context", "question", "response")
DEFAULT_INPUT_DIR = Path("data/raw/indian-legal-supervised-fine-tuning-data/data")
DEFAULT_REVISION = "fe58e87d8393ddd0b8e4224e3eec086c37f1f78e"
EXAMPLE_LIMIT_PER_OUTCOME = 5
EXAMPLE_LIMIT_PER_REASON = 3
PREVIEW_CHARS = 240
OUTCOMES = ("PASS", "QUARANTINE", "REJECT")


def list_shards(input_dir: Path) -> list[Path]:
    shards = sorted(input_dir.glob("*.parquet"), key=lambda item: item.name)
    if not shards:
        raise FileNotFoundError(f"No Parquet shards found in {input_dir}")
    return shards


def _shard_seed(seed: int, shard_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{shard_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def iter_shard_records(
    shard_path: Path,
    batch_size: int,
) -> Iterable[dict[str, Any]]:
    """Yield bounded context/q/a batches and 1-based row positions."""
    parquet = pq.ParquetFile(shard_path)
    row_number = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(REQUIRED_COLUMNS)):
        context_values = batch.column("context").to_pylist()
        question_values = batch.column("question").to_pylist()
        response_values = batch.column("response").to_pylist()
        for context, question, response in zip(context_values, question_values, response_values, strict=True):
            row_number += 1
            yield {
                "source_shard": shard_path.name,
                "source_row_number_1_based": row_number,
                "context": context,
                "question": question,
                "response": response,
            }
    if row_number != parquet.metadata.num_rows:
        raise RuntimeError(
            f"Read {row_number} rows from {shard_path.name}; metadata says {parquet.metadata.num_rows}"
        )


def reservoir_sample_shard(
    shard_path: Path,
    sample_size: int,
    *,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Select a deterministic uniform reservoir without preferring early rows."""
    rng = random.Random(_shard_seed(seed, shard_path.name))
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for record in iter_shard_records(shard_path, batch_size):
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(record)
        else:
            replacement = rng.randrange(seen)
            if replacement < sample_size:
                reservoir[replacement] = record
    reservoir.sort(key=lambda item: item["source_row_number_1_based"])
    return reservoir, seen


def _preview(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value[:PREVIEW_CHARS]


def _record_examples(
    outcome: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_row_id": record["source_row_id"],
        "source_shard": record["source_shard"],
        "source_row_number_1_based": record["source_row_number_1_based"],
        "outcome": outcome,
        "reason_code": record["reason_code"],
        "question_language": record["question_prediction"].get("language"),
        "response_language": record["response_prediction"].get("language"),
        "secondary_reason_codes": record.get("secondary_reason_codes", []),
        "quality_flags": record.get("quality_flags", []),
        "duplicate_of_source_row_id": record.get("duplicate_of_source_row_id"),
        "question_preview": _preview(record.get("question")),
        "response_preview": _preview(record.get("response")),
    }


def _prepare_output_directory(output_dir: Path, input_dir: Path) -> None:
    output_resolved = output_dir.resolve()
    input_resolved = input_dir.resolve()
    if output_resolved == input_resolved or input_resolved in output_resolved.parents:
        raise ValueError("Output directory must not be inside the raw input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(shards: list[Path], *, hash_files: bool) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        manifest[shard.name] = {
            "rows": int(parquet.metadata.num_rows),
            "bytes": shard.stat().st_size,
            "sha256": _sha256_file(shard) if hash_files else None,
        }
    return manifest


def _fingerprint_text(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _raw_text_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _open_duplicate_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute(
        "CREATE TABLE record_fingerprints (fingerprint BLOB PRIMARY KEY, canonical_source_row_id TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE duplicate_groups (fingerprint BLOB PRIMARY KEY) WITHOUT ROWID"
    )
    connection.commit()
    return connection


def _claim_fingerprint(
    connection: sqlite3.Connection,
    fingerprint: str,
    source_row_id: str,
) -> str | None:
    blob = bytes.fromhex(fingerprint)
    cursor = connection.execute(
        "INSERT OR IGNORE INTO record_fingerprints(fingerprint, canonical_source_row_id) VALUES (?, ?)",
        (blob, source_row_id),
    )
    if cursor.rowcount == 1:
        return None
    canonical = connection.execute(
        "SELECT canonical_source_row_id FROM record_fingerprints WHERE fingerprint=?",
        (blob,),
    ).fetchone()
    connection.execute(
        "INSERT OR IGNORE INTO duplicate_groups(fingerprint) VALUES (?)", (blob,)
    )
    return str(canonical[0]) if canonical else None


def _matched_missing_reference_fields(record: Mapping[str, Any]) -> list[str]:
    return [
        field
        for field in REQUIRED_COLUMNS
        if isinstance(record.get(field), str)
        and MISSING_LEGAL_REFERENCE_NUMBER.search(record[field])
    ]


def _quality_reason_codes(
    flags: list[str], missing_reference_fields: list[str]
) -> list[str]:
    codes: set[str] = set()
    if "placeholder_response" in flags:
        codes.add("PLACEHOLDER_RESPONSE_REVIEW")
    if "possible_context_dependent_question" in flags:
        codes.add("CONTEXT_DEPENDENT_QUESTION_REVIEW")
    if "statement_like_question" in flags:
        codes.add("STATEMENT_LIKE_QUESTION_REVIEW")
    for field in ("question", "response"):
        if f"control_characters_{field}" in flags:
            codes.add(f"CONTROL_CHARACTERS_{field.upper()}_REVIEW")
        if f"replacement_character_{field}" in flags:
            codes.add(f"REPLACEMENT_CHARACTER_{field.upper()}_REVIEW")
    if "response" in missing_reference_fields:
        codes.add("MISSING_LEGAL_REFERENCE_RESPONSE_REVIEW")
    if "question" in missing_reference_fields:
        codes.add("MISSING_LEGAL_REFERENCE_QUESTION_REVIEW")
    if "long_response" in flags:
        codes.add("LONG_RESPONSE_REVIEW")
    if "non_string_question" in flags:
        codes.add("NON_STRING_QUESTION_REVIEW")
    if "non_string_response" in flags:
        codes.add("NON_STRING_RESPONSE_REVIEW")
    return sorted(codes)


def _decision_from_language_and_quality(
    language_decision: Mapping[str, Any],
    quality_flags: list[str],
    quality_reason_codes: list[str],
    duplicate_of_source_row_id: str | None,
) -> tuple[str, str, str, list[str]]:
    """Apply fixed Stage 3 precedence and return outcome, reason, text, reasons."""
    if duplicate_of_source_row_id:
        return (
            "REJECT",
            "DUPLICATE_EXACT_FULL_RECORD",
            "Redundant normalized exact full-record occurrence; canonical row is retained.",
            ["DUPLICATE_EXACT_FULL_RECORD"],
        )

    missing_or_blank = []
    for field in ("question", "response"):
        if any(flag in quality_flags for flag in (f"missing_{field}", f"blank_{field}")):
            missing_or_blank.append(field)
    if missing_or_blank:
        names = "_AND_".join(field.upper() for field in missing_or_blank)
        return (
            "REJECT",
            f"MISSING_OR_BLANK_{names}",
            "A supervised record is missing a nonblank question or response.",
            [f"MISSING_OR_BLANK_{field.upper()}" for field in missing_or_blank],
        )

    quarantine_reasons = list(quality_reason_codes)
    if language_decision.get("outcome") != "PASS":
        quarantine_reasons.append(str(language_decision.get("reason_code", "LANGUAGE_UNCERTAIN")))
    if quarantine_reasons:
        ordered = [
            str(language_decision.get("reason_code", "")),
            "PLACEHOLDER_RESPONSE_REVIEW",
            "CONTEXT_DEPENDENT_QUESTION_REVIEW",
            "STATEMENT_LIKE_QUESTION_REVIEW",
            "CONTROL_CHARACTERS_QUESTION_REVIEW",
            "CONTROL_CHARACTERS_RESPONSE_REVIEW",
            "REPLACEMENT_CHARACTER_QUESTION_REVIEW",
            "REPLACEMENT_CHARACTER_RESPONSE_REVIEW",
            "MISSING_LEGAL_REFERENCE_QUESTION_REVIEW",
            "MISSING_LEGAL_REFERENCE_RESPONSE_REVIEW",
            "LONG_RESPONSE_REVIEW",
            "NON_STRING_QUESTION_REVIEW",
            "NON_STRING_RESPONSE_REVIEW",
        ]
        unique = set(quarantine_reasons)
        primary = next((code for code in ordered if code and code in unique), sorted(unique)[0])
        return (
            "QUARANTINE",
            primary,
            "At least one approved language or quality review trigger remains unresolved.",
            sorted(set(quarantine_reasons)),
        )

    return (
        "PASS",
        str(language_decision.get("reason_code", "APPROVED_CHECKS_PASSED")),
        "The record passed the approved deterministic Stage 3 checks; this is not legal or semantic certification.",
        [],
    )


def _stage3_parquet_schema() -> pa.Schema:
    fields = [
        pa.field("source_row_id", pa.string()),
        pa.field("dataset_revision", pa.string()),
        pa.field("source_shard", pa.string()),
        pa.field("source_row_number_1_based", pa.int64()),
        pa.field("outcome", pa.string()),
        pa.field("reason_code", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("secondary_reason_codes", pa.list_(pa.string())),
        pa.field("language_outcome", pa.string()),
        pa.field("language_reason_code", pa.string()),
        pa.field("language_secondary_reason_codes", pa.list_(pa.string())),
        pa.field("question", pa.string()),
        pa.field("response", pa.string()),
        pa.field("question_raw_sha256", pa.string()),
        pa.field("response_raw_sha256", pa.string()),
        pa.field("context_sha256", pa.string()),
        pa.field("record_fingerprint", pa.string()),
        pa.field("duplicate_of_source_row_id", pa.string()),
        pa.field("quality_flags", pa.list_(pa.string())),
        pa.field("quality_reason_codes", pa.list_(pa.string())),
        pa.field("missing_legal_reference_fields", pa.list_(pa.string())),
        pa.field("quality_signal_details_json", pa.string()),
        pa.field("sampling_selection_type", pa.string()),
        pa.field("sampling_seed", pa.int64()),
        pa.field("within_shard_inclusion_probability", pa.float64()),
        pa.field("context_use", pa.string()),
    ]
    for field in ("question", "response"):
        fields.extend(
            [
                pa.field(f"{field}_language", pa.string()),
                pa.field(f"{field}_language_normalized", pa.string()),
                pa.field(f"{field}_prediction_status", pa.string()),
                pa.field(f"{field}_classifier_score", pa.float64()),
                pa.field(f"{field}_score_is_probability", pa.bool_()),
                pa.field(f"{field}_input_characters", pa.int64()),
                pa.field(f"{field}_classified_characters", pa.int64()),
                pa.field(f"{field}_input_truncated", pa.bool_()),
                pa.field(f"{field}_prediction_error_type", pa.string()),
            ]
        )
    for field in REQUIRED_COLUMNS:
        fields.extend(
            [
                pa.field(f"{field}_form_feed_occurrences", pa.int64()),
                pa.field(f"{field}_control_character_occurrences", pa.int64()),
                pa.field(f"{field}_replacement_character_occurrences", pa.int64()),
            ]
        )
    return pa.schema(fields)


def _arrow_value(value: Any, expected: type | tuple[type, ...]) -> Any:
    if isinstance(value, bool):
        return None
    if isinstance(value, expected):
        return value
    return None


def _partition_row(record: Mapping[str, Any]) -> dict[str, Any]:
    language_decision = record.get("language_decision") or {}
    details = record.get("quality_signals") or {}
    sampling = record.get("sampling_metadata") or {}
    result = {
        key: record.get(key)
        for key in (
            "source_row_id",
            "dataset_revision",
            "source_shard",
            "source_row_number_1_based",
            "outcome",
            "reason_code",
            "reason",
            "secondary_reason_codes",
            "question",
            "response",
            "context_sha256",
            "record_fingerprint",
            "duplicate_of_source_row_id",
            "quality_flags",
            "quality_reason_codes",
            "missing_legal_reference_fields",
            "context_use",
        )
    }
    result.update(
        {
            "question_raw_sha256": _raw_text_sha256(record.get("question")),
            "response_raw_sha256": _raw_text_sha256(record.get("response")),
            "language_outcome": language_decision.get("outcome"),
            "language_reason_code": language_decision.get("reason_code"),
            "language_secondary_reason_codes": language_decision.get("secondary_reason_codes", []),
            "quality_signal_details_json": json.dumps(details, ensure_ascii=False, separators=(",", ":")),
            "sampling_selection_type": sampling.get("selection_type"),
            "sampling_seed": sampling.get("seed"),
            "within_shard_inclusion_probability": sampling.get("within_shard_inclusion_probability"),
        }
    )
    predictions = {
        "question": record.get("question_prediction") or {},
        "response": record.get("response_prediction") or {},
    }
    for field, prediction in predictions.items():
        result.update(
            {
                f"{field}_language": prediction.get("language"),
                f"{field}_language_normalized": normalize_language_code(prediction.get("language")),
                f"{field}_prediction_status": prediction.get("status"),
                f"{field}_classifier_score": _arrow_value(prediction.get("classifier_score"), (int, float)),
                f"{field}_score_is_probability": prediction.get("score_is_probability")
                if isinstance(prediction.get("score_is_probability"), bool)
                else None,
                f"{field}_input_characters": _arrow_value(prediction.get("input_characters"), int),
                f"{field}_classified_characters": _arrow_value(prediction.get("classified_characters"), int),
                f"{field}_input_truncated": prediction.get("input_truncated")
                if isinstance(prediction.get("input_truncated"), bool)
                else None,
                f"{field}_prediction_error_type": prediction.get("error_type"),
            }
        )
    form_feeds = details.get("form_feed_occurrences", {})
    controls = details.get("control_character_occurrences", {})
    replacements = details.get("replacement_character_occurrences", {})
    for field in REQUIRED_COLUMNS:
        result[f"{field}_form_feed_occurrences"] = int(form_feeds.get(field, 0))
        result[f"{field}_control_character_occurrences"] = int(controls.get(field, 0))
        result[f"{field}_replacement_character_occurrences"] = int(replacements.get(field, 0))
    for field in ("question", "response"):
        value = result[field]
        if value is not None and not isinstance(value, str):
            result[field] = str(value)
    return result


def _ledger_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Compact per-row provenance; source text remains only in outcome Parquet."""
    return {key: value for key, value in record.items() if key not in {"question", "response"}}


def _write_stage3_report(summary: Mapping[str, Any], report_path: Path) -> None:
    total = int(summary["processed_record_count"])
    lines = [
        "# LegalBrain AI Stage 3 cleaning report",
        "",
        f"- Dataset revision: `{summary['dataset_revision']}`",
        f"- Mode: **{summary['mode']}**; records processed: **{total:,}**",
        f"- All source shards fully processed: **{summary['all_shards_fully_processed']}**",
        f"- Source row reconciliation passed: **{summary['processed_rows_reconcile']}**",
        f"- Raw source hashes unchanged: **{summary.get('raw_source_unchanged', 'not checked in smoke mode')}**",
        f"- Peak resident memory: **{summary.get('peak_rss_bytes', 'unavailable')} bytes**",
        "",
        "## Mutually exclusive outcomes",
        "",
        "| Outcome | Rows | Percent |",
        "|---|---:|---:|",
    ]
    for outcome in OUTCOMES:
        count = int(summary["outcome_counts"].get(outcome, 0))
        percent = 100 * count / total if total else 0.0
        lines.append(f"| {outcome} | {count:,} | {percent:.4f}% |")
    lines.extend(["", "## Primary reason codes", "", "| Reason code | Rows |", "|---|---:|"])
    for code, count in sorted(summary["primary_reason_code_counts"].items(), key=lambda pair: (-pair[1], pair[0])):
        lines.append(f"| `{code}` | {count:,} |")
    lines.extend(["", "Secondary flags overlap and must not be summed as unique records:", ""])
    for code, count in sorted(summary["secondary_reason_code_counts"].items(), key=lambda pair: (-pair[1], pair[0])):
        lines.append(f"- `{code}`: {count:,}")
    lines.extend(["", "## Overlapping audit flags", ""])
    for code, count in sorted(summary["quality_flag_counts_overlapping"].items(), key=lambda pair: (-pair[1], pair[0])):
        lines.append(f"- `{code}`: {count:,}")
    lines.extend(["", "Missing-reference regex matches by field (review signals, not confirmed defects):", ""])
    for field, count in summary["missing_legal_reference_match_field_counts"].items():
        lines.append(f"- `{field}`: {count:,}")
    lines.extend(["", "## Classifier observations", "", "Raw output labels are classifier observations, not ground truth.", ""])
    for field in ("question", "response"):
        lines.append(f"- **{field} label counts:** `{summary['per_field_prediction_language_counts'][field]}`")
        lines.append(f"- **{field} categories:** `{summary['per_field_prediction_categories'][field]}`")
    lines.extend(["", "## Exact duplicate handling", ""])
    lines.append(
        f"- Redundant occurrences rejected: **{summary['duplicate_occurrence_count']:,}** across **{summary['duplicate_group_count']:,}** duplicate groups."
    )
    lines.append(f"- Unique normalized full-record fingerprints: **{summary['unique_full_record_count']:,}**.")
    for example in summary.get("duplicate_examples", []):
        lines.append(
            f"- `{example['duplicate_source_row_id']}` → canonical `{example['canonical_source_row_id']}`."
        )
    lines.extend(["", "## Representative records by outcome", ""])
    for outcome in OUTCOMES:
        lines.extend([f"### {outcome}", ""])
        for example in summary["representative_examples"].get(outcome, []):
            lines.append(
                f"- `{example['source_row_id']}` — `{example['reason_code']}`; "
                f"question: {(example.get('question_preview') or '')[:100]!r}; "
                f"response: {(example.get('response_preview') or '')[:100]!r}."
            )
        if not summary["representative_examples"].get(outcome):
            lines.append("- No records in this outcome.")
        lines.append("")
    lines.extend(["## Source field integrity statistics", ""])
    lines.append("Question/response text was copied without normalization; output length/null/blank/control/form-feed/U+FFFD statistics are reconciled to source.")
    lines.append("")
    lines.append("| Field | String | Null | Blank | Control values | Form-feed values | Form-feed occurrences | U+FFFD values | Mean chars |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for field in REQUIRED_COLUMNS:
        stats = summary["input_field_statistics"][field]
        lines.append(
            f"| {field} | {stats['string_values']:,} | {stats['null_values']:,} | {stats['blank_values']:,} | "
            f"{stats['values_with_control_characters']:,} | {stats['values_with_form_feed_page_break']:,} | "
            f"{stats['form_feed_page_break_occurrences']:,} | {stats['values_with_replacement_character']:,} | {stats['mean_characters']} |"
        )
    lines.extend([""])
    lines.extend(
        [
            "## Integrity and limitations",
            "",
            "- Form-feed U+000C page boundaries are preserved and are not treated as control corruption.",
            "- Question and response text is copied without normalization or repair; context is stored only as a fingerprint and used for audit flags and exact duplicate detection.",
            "- The label-based language policy is an operational classifier rule, not a calibrated guarantee of language identity.",
            "- PASS is not a legal-correctness or per-record semantic-answerability certification. Source-answer support is not fully adjudicated in this pass.",
            "- Short-response and context-only/legal-reference signals are retained for review and do not independently reject a record.",
            "",
        ]
    )
    tests = summary.get("test_results")
    if tests:
        lines.extend(["## Tests", ""])
        if isinstance(tests, Mapping):
            lines.append(
                f"- Unit tests: **{tests.get('passed', 'unknown')} passed; "
                f"{tests.get('failed', 'unknown')} failed** ({tests.get('tests_run', 'unknown')} run)."
            )
            if tests.get("command"):
                lines.append(f"- Command: `{tests['command']}`")
            if tests.get("syntax_check_passed") is not None:
                lines.append(
                    f"- Python syntax check: **{'passed' if tests['syntax_check_passed'] else 'failed'}**."
                )
            if tests.get("run_timing"):
                lines.append(f"- Timing: {tests['run_timing']}")
        else:
            lines.append(f"- {tests}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _measure_output_partitions(
    parquet_paths: Mapping[str, list[Path]],
) -> tuple[dict[str, int], dict[str, dict[str, Any]], dict[str, Any]]:
    row_counts = {outcome: 0 for outcome in OUTCOMES}
    stats = {field: new_field_stats() for field in ("question", "response")}
    exact_text_check = {"records_checked": 0, "question_hash_mismatches": 0, "response_hash_mismatches": 0}
    columns = [
        "question",
        "response",
        "question_raw_sha256",
        "response_raw_sha256",
    ]
    for outcome in OUTCOMES:
        for path in parquet_paths.get(outcome, []):
            parquet = pq.ParquetFile(path)
            row_counts[outcome] += int(parquet.metadata.num_rows)
            for batch in parquet.iter_batches(batch_size=4_096, columns=columns):
                question_values = batch.column("question").to_pylist()
                response_values = batch.column("response").to_pylist()
                question_hashes = batch.column("question_raw_sha256").to_pylist()
                response_hashes = batch.column("response_raw_sha256").to_pylist()
                for question, response, question_hash, response_hash in zip(
                    question_values,
                    response_values,
                    question_hashes,
                    response_hashes,
                    strict=True,
                ):
                    observe_field(stats["question"], question)
                    observe_field(stats["response"], response)
                    exact_text_check["records_checked"] += 1
                    if _raw_text_sha256(question) != question_hash:
                        exact_text_check["question_hash_mismatches"] += 1
                    if _raw_text_sha256(response) != response_hash:
                        exact_text_check["response_hash_mismatches"] += 1
    return (
        row_counts,
        {field: finalize_field_stats(values) for field, values in stats.items()},
        exact_text_check,
    )


def _peak_rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (AttributeError, OSError, ValueError):  # pragma: no cover - platform-specific
        return None


def _classify_and_decide(
    record: Mapping[str, Any],
    *,
    policy: LanguagePolicy,
    dataset_revision: str,
    classifier: Any,
    classifier_version: str,
    sampling_metadata: Mapping[str, Any],
    duplicate_of_source_row_id: str | None,
) -> dict[str, Any]:
    # Context is used only for audit flags and fingerprints, never for language.
    question_prediction = classify_field(record.get("question"), classifier=classifier)
    response_prediction = classify_field(record.get("response"), classifier=classifier)
    language_decision = decide_language_outcome(
        question_prediction,
        response_prediction,
        policy,
    )
    quality_flags = get_record_flags(dict(record))
    missing_reference_fields = _matched_missing_reference_fields(record)
    quality_reason_codes = _quality_reason_codes(quality_flags, missing_reference_fields)
    outcome, reason_code, reason, final_reasons = _decision_from_language_and_quality(
        language_decision,
        quality_flags,
        quality_reason_codes,
        duplicate_of_source_row_id,
    )
    context = record.get("context")
    context_fingerprint = _raw_text_sha256(context)
    full_fingerprint = stable_hash(
        [_fingerprint_text(record.get(field)) for field in REQUIRED_COLUMNS]
    )
    signal_details = {
        "audit_flags": quality_flags,
        "missing_legal_reference_fields": missing_reference_fields,
        "form_feed_occurrences": {
            field: record.get(field).count(FORM_FEED)
            if isinstance(record.get(field), str)
            else 0
            for field in REQUIRED_COLUMNS
        },
        "control_character_occurrences": {
            field: len(CONTROL_CHARACTERS.findall(record.get(field)))
            if isinstance(record.get(field), str)
            else 0
            for field in REQUIRED_COLUMNS
        },
        "replacement_character_occurrences": {
            field: record.get(field).count("\ufffd")
            if isinstance(record.get(field), str)
            else 0
            for field in REQUIRED_COLUMNS
        },
    }
    result = build_language_outcome_record(
        dataset_revision=dataset_revision,
        source_shard=str(record["source_shard"]),
        source_row_number_1_based=int(record["source_row_number_1_based"]),
        question=record.get("question"),
        response=record.get("response"),
        question_prediction=question_prediction,
        response_prediction=response_prediction,
        decision=language_decision,
        classifier_version=classifier_version,
        sampling_metadata=sampling_metadata,
        quality_signals=signal_details,
        context_sha256=context_fingerprint,
    )
    result.update(
        {
            "outcome": outcome,
            "outcome_scope": "stage3_record_quality_and_language",
            "reason_code": reason_code,
            "reason": reason,
            "secondary_reason_codes": sorted(
                set(final_reasons)
                | set(language_decision.get("secondary_reason_codes", []))
                | {f"AUDIT_FLAG_{flag.upper()}" for flag in quality_flags if flag != reason_code.lower()}
            ),
            "language_decision": dict(language_decision),
            "quality_flags": quality_flags,
            "quality_reason_codes": quality_reason_codes,
            "missing_legal_reference_fields": missing_reference_fields,
            "record_fingerprint": full_fingerprint,
            "duplicate_of_source_row_id": duplicate_of_source_row_id,
            "context_sha256": context_fingerprint,
            "context_use": "read_for_provenance_and_quality_flags_only; never used for language classification",
        }
    )
    return result


def run_language_cleaning(
    *,
    input_dir: Path,
    output_dir: Path,
    dataset_revision: str = DEFAULT_REVISION,
    policy: LanguagePolicy | None = None,
    batch_size: int = 2_048,
    sample_per_shard: int | None = None,
    seed: int = 20261029,
    classifier: Any | None = None,
) -> dict[str, Any]:
    """Run Stage 3 rules with bounded Arrow batches and a disk-backed dedupe index."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    sample_mode = sample_per_shard is not None
    if sample_mode and sample_per_shard is not None and sample_per_shard < 1:
        raise ValueError("sample_per_shard must be at least 1")
    effective_policy = policy or LanguagePolicy()
    if not sample_mode and not effective_policy.is_operational:
        raise RuntimeError(
            "Full-corpus Stage 3 is blocked until an approved operational language policy is supplied."
        )
    if effective_policy.is_operational and effective_policy.decision_mode != "label_based":
        raise RuntimeError("This Stage 3 V1 run requires the approved label_based policy; score floors are not used.")

    shards = list_shards(input_dir)
    metadata_rows: dict[str, int] = {}
    question_response_compressed_bytes = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        missing_columns = set(REQUIRED_COLUMNS) - set(parquet.schema.names)
        if missing_columns:
            raise ValueError(f"{shard.name} is missing columns: {sorted(missing_columns)}")
        metadata_rows[shard.name] = int(parquet.metadata.num_rows)
        for index, column_name in enumerate(parquet.schema.names):
            if column_name in {"question", "response"}:
                question_response_compressed_bytes += sum(
                    int(parquet.metadata.row_group(group).column(index).total_compressed_size)
                    for group in range(parquet.metadata.num_row_groups)
                )
    expected_rows = sum(metadata_rows.values())
    if not sample_mode and dataset_revision == DEFAULT_REVISION:
        if len(shards) != 33 or expected_rows != 6_055_371:
            raise RuntimeError(
                f"Expected 33 shards and 6,055,371 metadata rows for {DEFAULT_REVISION}; "
                f"found {len(shards)} shards and {expected_rows:,} rows."
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    available_bytes = shutil.disk_usage(output_dir.parent).free
    estimated_minimum_free_bytes = max(
        8 * 1024**3,
        3 * question_response_compressed_bytes + 2 * 1024**3,
    )
    if not sample_mode and available_bytes < estimated_minimum_free_bytes:
        raise RuntimeError(
            f"Insufficient free disk space: {available_bytes:,} bytes available; "
            f"estimated minimum is {estimated_minimum_free_bytes:,} bytes."
        )
    _prepare_output_directory(output_dir, input_dir)

    policy_used_path = output_dir / "language_policy_used.json"
    policy_used = {
        "language_gate_approved": effective_policy.language_gate_approved,
        "classifier": effective_policy.classifier,
        "decision_mode": effective_policy.decision_mode,
        "validation_reference": effective_policy.validation_reference,
        "approved_by": effective_policy.approved_by,
        "minimum_classifier_score_by_field": dict(
            effective_policy.minimum_classifier_score_by_field or {}
        ),
    }
    policy_used_path.write_text(
        json.dumps(policy_used, indent=2) + "\n", encoding="utf-8"
    )
    policy_used_sha256 = _sha256_file(policy_used_path)

    source_manifest_before = _source_manifest(shards, hash_files=not sample_mode)
    if not sample_mode:
        (output_dir / "source_manifest_before.json").write_text(
            json.dumps(source_manifest_before, indent=2) + "\n", encoding="utf-8"
        )

    if classifier is None:
        try:
            import langid as classifier_module
        except ImportError as exc:  # pragma: no cover - dependency is in project config
            raise RuntimeError("langid is required for the Stage 3 language gate") from exc
        classifier = classifier_module
    classifier_version = language_classifier_version()

    processed = 0
    outcome_counts: Counter[str] = Counter({outcome: 0 for outcome in OUTCOMES})
    primary_reason_counts: Counter[str] = Counter()
    secondary_reason_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    quality_reason_counts: Counter[str] = Counter()
    language_decision_counts: Counter[str] = Counter()
    prediction_counts: dict[str, Counter[str]] = {
        "question": Counter(),
        "response": Counter(),
    }
    prediction_status_counts: dict[str, Counter[str]] = {
        "question": Counter(),
        "response": Counter(),
    }
    text_integrity_occurrences = {
        field: Counter({"form_feed": 0, "control_characters": 0, "replacement_character": 0})
        for field in REQUIRED_COLUMNS
    }
    field_stats_input = {field: new_field_stats() for field in REQUIRED_COLUMNS}
    shard_counts: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[dict[str, Any]]] = {
        "PASS": [],
        "QUARANTINE": [],
        "REJECT": [],
    }
    examples_seen: Counter[str] = Counter()
    example_rngs = {
        outcome: random.Random(_shard_seed(seed, f"examples:{outcome}"))
        for outcome in examples
    }
    duplicate_occurrence_count = 0
    duplicate_examples: list[dict[str, Any]] = []
    parquet_paths: dict[str, list[Path]] = {outcome: [] for outcome in OUTCOMES}
    partition_schema = _stage3_parquet_schema()
    partition_root = output_dir / "outcome_partitions"
    for outcome in OUTCOMES:
        (partition_root / outcome).mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "fingerprint_index.sqlite3"
    connection = _open_duplicate_index(index_path)

    ledger_path = output_dir / "outcome_ledger.jsonl.gz"
    try:
        with ledger_path.open("wb") as raw_ledger:
            with gzip.GzipFile(
                filename="", fileobj=raw_ledger, mode="wb", compresslevel=1, mtime=0
            ) as ledger:
                for shard_index, shard in enumerate(shards, start=1):
                    metadata_count = metadata_rows[shard.name]
                    if sample_mode:
                        selected, actual_shard_rows = reservoir_sample_shard(
                            shard,
                            int(sample_per_shard),
                            batch_size=batch_size,
                            seed=seed,
                        )
                        rows: Iterable[dict[str, Any]] = selected
                        selection_type = "uniform_per_shard_reservoir"
                    else:
                        actual_shard_rows = metadata_count
                        rows = iter_shard_records(shard, batch_size)
                        selection_type = "full_scan"

                    selected_count = len(rows) if sample_mode else actual_shard_rows
                    shard_counts[shard.name] = {
                        "metadata_rows": metadata_count,
                        "source_rows_read_for_selection": actual_shard_rows,
                        "selected_rows": selected_count,
                        "processed_rows": 0,
                        "outcomes": {outcome: 0 for outcome in OUTCOMES},
                        "primary_reason_code_counts": {},
                    }
                    shard_primary_reasons: Counter[str] = Counter()
                    shard_writers: dict[str, pq.ParquetWriter] = {}
                    buffered: dict[str, list[dict[str, Any]]] = {outcome: [] for outcome in OUTCOMES}
                    sampled_probability = (
                        min(1.0, selected_count / actual_shard_rows)
                        if actual_shard_rows
                        else 0.0
                    )

                    def flush_buffers() -> None:
                        for outcome in OUTCOMES:
                            records = buffered[outcome]
                            if not records:
                                continue
                            if outcome not in shard_writers:
                                path = partition_root / outcome / f"{shard.stem}.parquet"
                                shard_writers[outcome] = pq.ParquetWriter(
                                    path,
                                    partition_schema,
                                    compression="zstd",
                                    use_dictionary=True,
                                )
                                parquet_paths[outcome].append(path)
                            table = pa.Table.from_pylist(records, schema=partition_schema)
                            shard_writers[outcome].write_table(table, row_group_size=len(records))
                            records.clear()
                        connection.commit()

                    try:
                        for source_record in rows:
                            for field in REQUIRED_COLUMNS:
                                observe_field(field_stats_input[field], source_record.get(field))

                            source_row_id = (
                                f"{dataset_revision}:{source_record['source_shard']}:"
                                f"{source_record['source_row_number_1_based']}"
                            )
                            full_fingerprint = stable_hash(
                                [_fingerprint_text(source_record.get(field)) for field in REQUIRED_COLUMNS]
                            )
                            duplicate_of = _claim_fingerprint(
                                connection, full_fingerprint, source_row_id
                            )
                            sampling_metadata = {
                                "selection_type": selection_type,
                                "seed": seed if sample_mode else None,
                                "source_rows_in_shard": actual_shard_rows,
                                "selected_rows_in_shard": selected_count,
                                "within_shard_inclusion_probability": sampled_probability,
                            }
                            outcome_record = _classify_and_decide(
                                source_record,
                                policy=effective_policy,
                                dataset_revision=dataset_revision,
                                classifier=classifier,
                                classifier_version=classifier_version,
                                sampling_metadata=sampling_metadata,
                                duplicate_of_source_row_id=duplicate_of,
                            )

                            processed += 1
                            shard_counts[shard.name]["processed_rows"] += 1
                            outcome = outcome_record["outcome"]
                            outcome_counts[outcome] += 1
                            shard_counts[shard.name]["outcomes"][outcome] += 1
                            primary_reason = outcome_record["reason_code"]
                            primary_reason_counts[primary_reason] += 1
                            shard_primary_reasons[primary_reason] += 1
                            for code in set(outcome_record["secondary_reason_codes"]):
                                secondary_reason_counts[code] += 1
                            for flag in set(outcome_record["quality_flags"]):
                                quality_flag_counts[flag] += 1
                            for code in set(outcome_record["quality_reason_codes"]):
                                quality_reason_counts[code] += 1

                            language_decision_counts[str(outcome_record["language_decision"]["reason_code"])] += 1
                            for field in ("question", "response"):
                                prediction = outcome_record[f"{field}_prediction"]
                                language = prediction.get("language")
                                prediction_counts[field][str(language) if language else "<missing>"] += 1
                                status = prediction.get("status")
                                prediction_status_counts[field][str(status or "<missing>")] += 1
                            details = outcome_record["quality_signals"]
                            for field in REQUIRED_COLUMNS:
                                for signal, count in details["form_feed_occurrences"].items():
                                    if signal == field:
                                        text_integrity_occurrences[field]["form_feed"] += count
                                for signal, count in details["control_character_occurrences"].items():
                                    if signal == field:
                                        text_integrity_occurrences[field]["control_characters"] += count
                                for signal, count in details["replacement_character_occurrences"].items():
                                    if signal == field:
                                        text_integrity_occurrences[field]["replacement_character"] += count
                            for field in outcome_record["missing_legal_reference_fields"]:
                                secondary_reason_counts[f"MISSING_LEGAL_REFERENCE_MATCH_{field.upper()}"] += 1

                            ledger.write(
                                (
                                    json.dumps(_ledger_record(outcome_record), ensure_ascii=False, separators=(",", ":"), default=str)
                                    + "\n"
                                ).encode("utf-8")
                            )
                            buffered[outcome].append(_partition_row(outcome_record))

                            examples_seen[outcome] += 1
                            example = _record_examples(outcome, outcome_record)
                            if len(examples[outcome]) < EXAMPLE_LIMIT_PER_OUTCOME:
                                examples[outcome].append(example)
                            else:
                                replacement = example_rngs[outcome].randrange(examples_seen[outcome])
                                if replacement < EXAMPLE_LIMIT_PER_OUTCOME:
                                    examples[outcome][replacement] = example

                            if duplicate_of:
                                duplicate_occurrence_count += 1
                                if len(duplicate_examples) < 5:
                                    duplicate_examples.append(
                                        {
                                            "canonical_source_row_id": duplicate_of,
                                            "duplicate_source_row_id": source_row_id,
                                            "record_fingerprint": full_fingerprint,
                                            "source_shard": source_record["source_shard"],
                                            "source_row_number_1_based": source_record["source_row_number_1_based"],
                                            "question_preview": _preview(source_record.get("question")),
                                            "response_preview": _preview(source_record.get("response")),
                                        }
                                    )

                            if processed % batch_size == 0:
                                flush_buffers()
                            if not sample_mode and processed % 25_000 == 0:
                                print(f"Stage 3 processed {processed:,}/{expected_rows:,} rows", flush=True)
                        flush_buffers()
                    finally:
                        for writer in shard_writers.values():
                            writer.close()
                    shard_counts[shard.name]["primary_reason_code_counts"] = dict(shard_primary_reasons)
                    print(
                        f"[{shard_index}/{len(shards)}] {shard.name}: "
                        f"processed {shard_counts[shard.name]['processed_rows']:,} rows",
                        flush=True,
                    )
    finally:
        connection.commit()
        connection.close()

    expected_selected = (
        sum(min(int(sample_per_shard), rows) for rows in metadata_rows.values())
        if sample_mode
        else expected_rows
    )
    if processed != expected_selected:
        raise RuntimeError(
            f"Processed {processed} records but expected {expected_selected} from shard metadata"
        )
    if sum(outcome_counts.values()) != processed:
        raise RuntimeError("Outcome counts do not reconcile with processed records")

    with sqlite3.connect(index_path) as index_connection:
        duplicate_group_count = int(
            index_connection.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[0]
        )
        unique_full_record_count = int(
            index_connection.execute("SELECT COUNT(*) FROM record_fingerprints").fetchone()[0]
        )
    partition_row_counts, output_field_stats, raw_text_check = _measure_output_partitions(parquet_paths)
    if partition_row_counts != dict(outcome_counts):
        raise RuntimeError(
            f"Parquet outcome counts do not reconcile: {partition_row_counts} vs {dict(outcome_counts)}"
        )
    if raw_text_check["question_hash_mismatches"] or raw_text_check["response_hash_mismatches"]:
        raise RuntimeError(f"Raw question/response text changed in derived output: {raw_text_check}")
    input_field_stats = {
        field: finalize_field_stats(values) for field, values in field_stats_input.items()
    }
    q_response_stats_reconcile = all(
        input_field_stats[field] == output_field_stats[field]
        for field in ("question", "response")
    )

    ledger_lines = 0
    with gzip.open(ledger_path, "rt", encoding="utf-8") as ledger_check:
        for _ in ledger_check:
            ledger_lines += 1
    if ledger_lines != processed:
        raise RuntimeError(f"Outcome ledger has {ledger_lines} records; expected {processed}")

    source_manifest_after = None
    raw_source_unchanged: bool | None = None
    if not sample_mode:
        source_manifest_after = _source_manifest(shards, hash_files=True)
        (output_dir / "source_manifest_after.json").write_text(
            json.dumps(source_manifest_after, indent=2) + "\n", encoding="utf-8"
        )
        raw_source_unchanged = all(
            source_manifest_before[name]["sha256"] == source_manifest_after[name]["sha256"]
            and source_manifest_before[name]["bytes"] == source_manifest_after[name]["bytes"]
            for name in source_manifest_before
        )

    language_categories: dict[str, dict[str, int]] = {}
    for field, counts in prediction_counts.items():
        categories: Counter[str] = Counter()
        for raw_label, count in counts.items():
            normalized = normalize_language_code(raw_label)
            if normalized == "en":
                category = "english"
            elif normalized == "hi":
                category = "hindi"
            elif normalized == "mixed":
                category = "mixed"
            elif normalized == "uncertain":
                category = "uncertain"
            elif normalized is None:
                category = "missing_or_unknown"
            else:
                category = "unsupported"
            categories[category] += count
        language_categories[field] = dict(categories)

    output_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    per_shard_selection_count = sum(item["source_rows_read_for_selection"] for item in shard_counts.values())
    summary_path = output_dir / "stage3_summary.json"
    report_path = output_dir / "stage3_report.md"
    outcome_percentages = {
        outcome: outcome_counts[outcome] / processed * 100 if processed else 0.0
        for outcome in OUTCOMES
    }
    try:
        raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_rss_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
    except (AttributeError, OSError, ValueError):  # pragma: no cover - platform-specific
        peak_rss_bytes = None

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": dataset_revision,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "classifier": effective_policy.classifier,
        "classifier_version": classifier_version,
        "classifier_score_is_probability": False,
        "language_gate_approved": effective_policy.language_gate_approved,
        "language_gate_operational": effective_policy.is_operational,
        "language_decision_mode": effective_policy.decision_mode,
        "validation_reference": effective_policy.validation_reference,
        "approved_by": effective_policy.approved_by,
        "language_policy_used": policy_used,
        "language_policy_used_sha256": policy_used_sha256,
        "per_field_score_floors_configured": {
            field: value
            for field, value in (effective_policy.minimum_classifier_score_by_field or {}).items()
        },
        "outcome_scope": "full_stage3_record_quality_and_language",
        "mode": "sample" if sample_mode else "full_corpus",
        "sampling": {
            "method": "fixed-seed independent reservoir within each shard"
            if sample_mode
            else "all rows in each shard",
            "seed": seed if sample_mode else None,
            "requested_per_shard": sample_per_shard,
            "actual_record_count": processed,
            "sample_is_corpus_estimate": False if sample_mode else None,
        },
        "source_metadata_rows": expected_rows,
        "source_rows_streamed_for_selection": per_shard_selection_count,
        "processed_record_count": processed,
        "all_shards_fully_processed": not sample_mode,
        "processed_rows_reconcile": processed == expected_selected and processed == sum(outcome_counts.values()),
        "outcome_counts": {outcome: int(outcome_counts[outcome]) for outcome in OUTCOMES},
        "outcome_percentages_of_processed_records": outcome_percentages,
        "primary_reason_code_counts": dict(primary_reason_counts),
        "secondary_reason_code_counts": dict(secondary_reason_counts),
        "quality_flag_counts_overlapping": dict(quality_flag_counts),
        "quality_reason_code_counts_overlapping": dict(quality_reason_counts),
        "language_decision_reason_counts": dict(language_decision_counts),
        "per_field_prediction_language_counts": {
            field: dict(counts) for field, counts in prediction_counts.items()
        },
        "per_field_prediction_categories": language_categories,
        "per_field_prediction_status_counts": {
            field: dict(counts) for field, counts in prediction_status_counts.items()
        },
        "missing_legal_reference_match_field_counts": {
            field: secondary_reason_counts.get(f"MISSING_LEGAL_REFERENCE_MATCH_{field.upper()}", 0)
            for field in REQUIRED_COLUMNS
        },
        "text_integrity_occurrences": {
            field: dict(counts) for field, counts in text_integrity_occurrences.items()
        },
        "input_field_statistics": input_field_stats,
        "output_question_response_field_statistics": output_field_stats,
        "question_response_field_statistics_reconcile": q_response_stats_reconcile,
        "raw_question_response_text_hash_validation": raw_text_check,
        "transformations_applied": [],
        "per_shard": shard_counts,
        "representative_examples_seed": seed,
        "representative_examples": examples,
        "duplicate_occurrence_count": duplicate_occurrence_count,
        "duplicate_group_count": duplicate_group_count,
        "unique_full_record_count": unique_full_record_count,
        "duplicate_examples": duplicate_examples,
        "fingerprint_index": {
            "path": index_path.name,
            "storage": "SQLite disk-backed database",
            "cache_size_bytes": 32 * 1024 * 1024,
            "temporary_storage": "file",
            "fingerprint_normalization": "NFC Unicode normalization and whitespace collapse over context, question, response",
            "canonical_order": "first row in sorted shard filename then ascending 1-based source row",
        },
        "source_manifest_before": source_manifest_before if not sample_mode else None,
        "source_manifest_after": source_manifest_after,
        "raw_source_unchanged": raw_source_unchanged,
        "context_read_or_used_for_language": False,
        "context_stored_as_raw_text": False,
        "context_fingerprints_for_provenance": True,
        "output_file_counts": {
            outcome: partition_row_counts[outcome] for outcome in OUTCOMES
        },
        "outcome_ledger_record_count": ledger_lines,
        "outcome_ledger_reconciles": ledger_lines == processed,
        "available_disk_bytes_at_start": available_bytes,
        "estimated_minimum_free_disk_bytes": estimated_minimum_free_bytes,
        "output_bytes": output_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "memory_model": {
            "arrow_batch_size": batch_size,
            "fingerprints_loaded_into_python_memory": False,
            "fingerprint_index_path": index_path.name,
            "fingerprint_index_disk_backed": True,
            "full_source_text_buffered_beyond_batch": False,
        },
        "output_files": {
            "pass_parquet_directory": "outcome_partitions/PASS",
            "quarantine_parquet_directory": "outcome_partitions/QUARANTINE",
            "reject_parquet_directory": "outcome_partitions/REJECT",
            "outcome_ledger": ledger_path.name,
            "summary": summary_path.name,
            "report": report_path.name,
            "language_policy_used": policy_used_path.name,
            "fingerprint_index": index_path.name,
        },
        "limitations": [
            "The language classifier is imperfect; the approved label-based policy is operational but is not a calibrated guarantee of language identity.",
            "Language predictions are not legal correctness or semantic answerability judgments.",
            "Question-only suitability and source-answer support are not fully adjudicated for every record; lexical signals route candidates to quarantine.",
            "Context-only control, replacement, and missing-reference matches are retained as provenance/review signals and do not by themselves block a self-contained question-answer record.",
            "Short responses are recorded as review signals and never rejected or quarantined solely by character length.",
            "Exact placeholder responses are quarantined pending a separately approved exact-template exclusion list.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_stage3_report(summary, report_path)
    if raw_source_unchanged is False:
        raise RuntimeError("Raw source checksums changed during Stage 3; inspect the saved report and manifests.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded-memory Stage 3 quality and English/Hindi outcome processing."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=20261029)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sample-per-shard",
        type=int,
        help="Uniform random reservoir sample size per shard for a Stage 3 smoke run.",
    )
    mode.add_argument(
        "--full-scan",
        action="store_true",
        help="Process every source row; requires the explicit approved label-based policy config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = LanguagePolicy.load(args.policy_config) if args.policy_config else LanguagePolicy()
    summary = run_language_cleaning(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_revision=args.dataset_revision,
        policy=policy,
        batch_size=args.batch_size,
        sample_per_shard=args.sample_per_shard,
        seed=args.seed,
    )
    print(
        f"Processed {summary['processed_record_count']} records; outcomes="
        f"{summary['outcome_counts']}; wrote {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover - tested through run_language_cleaning
    main()
