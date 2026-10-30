"""Audit multiple normalized responses per question in Stage 3 PASS data.

The utility reads only the derived PASS Parquet partitions. It uses the same
NFC-plus-whitespace normalization and stable hash as the existing audit. A
disk-backed SQLite work index keeps memory bounded; only multiple-response
groups and a bounded review sample are written to the report directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq

try:
    from scripts.data_pipeline.audit.run_audit import normalize_for_comparison, stable_hash
except ModuleNotFoundError:  # pragma: no cover - direct script path fallback
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.data_pipeline.audit.run_audit import normalize_for_comparison, stable_hash


DEFAULT_STAGE3_DIR = Path("data/processed/stage3-label-based-v1")
DEFAULT_OUTPUT_DIR = Path("data/reports/multiple-response-audit-v1")
DEFAULT_SEED = 20261030
DEFAULT_BATCH_SIZE = 2_048
DEFAULT_SAMPLE_SIZE = 100
OUTCOMES = ("PASS", "QUARANTINE", "REJECT")
PROVENANCE_COLUMNS = (
    "source_row_id",
    "dataset_revision",
    "source_shard",
    "source_row_number_1_based",
    "question",
    "response",
    "quality_flags",
    "reason_code",
    "secondary_reason_codes",
    "record_fingerprint",
    "context_sha256",
    "question_language",
    "question_language_normalized",
    "question_prediction_status",
    "question_classifier_score",
    "response_language",
    "response_language_normalized",
    "response_prediction_status",
    "response_classifier_score",
)


def _as_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"PASS row has a non-string {field_name}: {value!r}")
    return value


def question_group_id(question: str) -> str:
    """Stable ID compatible with the existing audit's question fingerprint."""
    return f"qgrp_{stable_hash([question])}"


def response_variant_id(response: str) -> str:
    """Stable response key using the existing audit normalization."""
    return stable_hash([response])


def _content_signature(value: str) -> str:
    """A sampling-only surface comparison; never used to count response variants."""
    normalized = normalize_for_comparison(value).casefold()
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _random_rank(seed: int, group_key: str) -> str:
    return hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).hexdigest()


def _open_work_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute(
        """CREATE TABLE question_groups (
            group_key TEXT PRIMARY KEY,
            normalized_question TEXT NOT NULL,
            question_text TEXT NOT NULL,
            group_size INTEGER NOT NULL
        ) WITHOUT ROWID"""
    )
    connection.execute(
        """CREATE TABLE response_variants (
            group_key TEXT NOT NULL,
            response_key TEXT NOT NULL,
            normalized_response TEXT NOT NULL,
            representative_response_text TEXT NOT NULL,
            normalized_response_characters INTEGER NOT NULL,
            content_signature TEXT NOT NULL,
            response_frequency INTEGER NOT NULL,
            PRIMARY KEY (group_key, response_key)
        ) WITHOUT ROWID"""
    )
    connection.commit()
    return connection


def _create_multiple_group_tables(connection: sqlite3.Connection, seed: int) -> None:
    connection.execute(
        """CREATE TABLE multiple_groups (
            group_key TEXT PRIMARY KEY,
            group_size INTEGER NOT NULL,
            distinct_response_count INTEGER NOT NULL,
            min_response_characters INTEGER NOT NULL,
            max_response_characters INTEGER NOT NULL,
            distinct_surface_forms INTEGER NOT NULL,
            random_rank TEXT NOT NULL,
            has_hindi_or_mixed INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID"""
    )
    cursor = connection.execute(
        """SELECT q.group_key, q.group_size, COUNT(*) AS response_count,
                  MIN(r.normalized_response_characters),
                  MAX(r.normalized_response_characters),
                  COUNT(DISTINCT r.content_signature)
           FROM question_groups q
           JOIN response_variants r USING (group_key)
           GROUP BY q.group_key
           HAVING COUNT(*) > 1"""
    )
    while True:
        rows = cursor.fetchmany(1_000)
        if not rows:
            break
        connection.executemany(
            """INSERT INTO multiple_groups (
                group_key, group_size, distinct_response_count,
                min_response_characters, max_response_characters,
                distinct_surface_forms, random_rank
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    group_key,
                    group_size,
                    response_count,
                    min_chars,
                    max_chars,
                    surface_forms,
                    _random_rank(seed, group_key),
                )
                for group_key, group_size, response_count, min_chars, max_chars, surface_forms in rows
            ],
        )
    connection.execute(
        """CREATE TABLE group_records (
            group_key TEXT NOT NULL,
            response_key TEXT NOT NULL,
            source_row_id TEXT NOT NULL,
            source_shard TEXT NOT NULL,
            source_row_number_1_based INTEGER NOT NULL,
            dataset_revision TEXT,
            question_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            quality_flags_json TEXT NOT NULL,
            reason_code TEXT,
            secondary_reason_codes_json TEXT NOT NULL,
            record_fingerprint TEXT,
            context_sha256 TEXT,
            question_language TEXT,
            question_language_normalized TEXT,
            question_prediction_status TEXT,
            question_classifier_score REAL,
            response_language TEXT,
            response_language_normalized TEXT,
            response_prediction_status TEXT,
            response_classifier_score REAL,
            PRIMARY KEY (group_key, response_key, source_row_id)
        ) WITHOUT ROWID"""
    )
    connection.execute(
        "CREATE INDEX group_records_order_idx ON group_records(group_key, response_key, source_shard, source_row_number_1_based)"
    )
    connection.commit()


def _stage3_input_manifest(stage3_dir: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    summary_path = stage3_dir / "stage3_summary.json"
    before_path = stage3_dir / "source_manifest_before.json"
    after_path = stage3_dir / "source_manifest_after.json"
    for path in (summary_path, before_path, after_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 3 provenance file is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    if summary.get("mode") != "full_corpus" or not summary.get("all_shards_fully_processed"):
        raise ValueError("Stage 3 summary does not certify a complete full-corpus run")
    if not summary.get("raw_source_unchanged"):
        raise ValueError("Stage 3 summary does not certify unchanged raw source hashes")
    if set(before) != set(after) or not before:
        raise ValueError("Stage 3 before/after source manifests have different shard sets")
    for name in before:
        if before[name].get("sha256") != after[name].get("sha256"):
            raise ValueError(f"Raw source hash changed for {name}")
        if before[name].get("bytes") != after[name].get("bytes"):
            raise ValueError(f"Raw source size changed for {name}")
    metadata_rows = sum(int(item["rows"]) for item in before.values())
    if metadata_rows != int(summary.get("source_metadata_rows", -1)):
        raise ValueError("Stage 3 source metadata rows do not reconcile with source manifests")
    if sum(int(summary.get("outcome_counts", {}).get(k, 0)) for k in OUTCOMES) != metadata_rows:
        raise ValueError("Stage 3 outcome counts do not reconcile with source metadata")
    return summary, before, metadata_rows


def _pass_parts(pass_dir: Path, expected_rows: int) -> tuple[list[Path], dict[str, int]]:
    if not pass_dir.is_dir():
        raise FileNotFoundError(f"Stage 3 PASS partition directory is missing: {pass_dir}")
    parts = sorted(pass_dir.glob("*.parquet"), key=lambda item: item.name)
    if not parts:
        raise FileNotFoundError(f"No PASS Parquet files found under {pass_dir}")
    counts: dict[str, int] = {}
    total = 0
    required = set(PROVENANCE_COLUMNS) | {"outcome"}
    for path in parts:
        parquet = pq.ParquetFile(path)
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path.name} is missing required PASS fields: {sorted(missing)}")
        counts[path.name] = int(parquet.metadata.num_rows)
        total += counts[path.name]
    if total != expected_rows:
        raise ValueError(f"PASS partition has {total:,} rows, expected {expected_rows:,}")
    return parts, counts


def _iter_pass_batches(parts: Sequence[Path], batch_size: int) -> Iterable[tuple[Path, Any]]:
    columns = list(PROVENANCE_COLUMNS) + ["outcome"]
    for path in parts:
        parquet = pq.ParquetFile(path)
        row_count = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            row_count += batch.num_rows
            yield path, batch
        if row_count != parquet.metadata.num_rows:
            raise RuntimeError(
                f"Read {row_count:,} rows from {path.name}; metadata says {parquet.metadata.num_rows:,}"
            )


def _record_key_values(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    question = _as_text(record.get("question"), field_name="question")
    response = _as_text(record.get("response"), field_name="response")
    normalized_question = normalize_for_comparison(question)
    normalized_response = normalize_for_comparison(response)
    if not normalized_question or not normalized_response:
        raise ValueError("Stage 3 PASS partition contains a blank question or response")
    return (
        stable_hash([question]),
        normalized_question,
        stable_hash([response]),
        normalized_response,
    )


def _language_label(record: Mapping[str, Any], field: str) -> str:
    value = record.get(f"{field}_language")
    return str(value) if isinstance(value, str) and value else "<missing>"


def _collect_first_pass(
    parts: Sequence[Path],
    connection: sqlite3.Connection,
    *,
    batch_size: int,
) -> dict[str, Any]:
    language_counts: dict[str, Counter[str]] = {
        "question": Counter(),
        "response": Counter(),
    }
    quality_flag_counts: Counter[str] = Counter()
    per_shard_rows: Counter[str] = Counter()
    processed = 0
    group_rows: list[tuple[str, str, str, int]] = []
    variant_rows: list[tuple[str, str, str, str, int, str, int]] = []
    next_progress_at = 250_000

    def flush() -> None:
        if group_rows:
            connection.executemany(
                """INSERT INTO question_groups(group_key, normalized_question, question_text, group_size)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(group_key) DO UPDATE SET group_size=question_groups.group_size+1
                   WHERE question_groups.normalized_question=excluded.normalized_question""",
                group_rows,
            )
            group_rows.clear()
        if variant_rows:
            connection.executemany(
                """INSERT INTO response_variants(
                    group_key, response_key, normalized_response, representative_response_text,
                    normalized_response_characters, content_signature, response_frequency
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_key, response_key) DO UPDATE SET
                    response_frequency=response_variants.response_frequency+1
                WHERE response_variants.normalized_response=excluded.normalized_response""",
                variant_rows,
            )
            variant_rows.clear()
        connection.commit()

    for path, batch in _iter_pass_batches(parts, batch_size):
        values = batch.to_pylist()
        for record in values:
            if record.get("outcome") != "PASS":
                raise ValueError(f"Non-PASS row found in PASS partition: {record.get('source_row_id')}")
            group_key, normalized_question, response_key, normalized_response = _record_key_values(record)
            question = _as_text(record["question"], field_name="question")
            response = _as_text(record["response"], field_name="response")
            group_rows.append((group_key, normalized_question, question, 1))
            variant_rows.append(
                (
                    group_key,
                    response_key,
                    normalized_response,
                    response,
                    len(normalized_response),
                    _content_signature(normalized_response),
                    1,
                )
            )
            per_shard_rows[path.name] += 1
            processed += 1
            for field in ("question", "response"):
                language_counts[field][_language_label(record, field)] += 1
            for flag in record.get("quality_flags") or []:
                quality_flag_counts[str(flag)] += 1
            if len(group_rows) >= batch_size:
                flush()
        flush()
        if processed >= next_progress_at:
            print(f"Indexed {processed:,} PASS records", flush=True)
            next_progress_at = ((processed // 250_000) + 1) * 250_000
    indexed_group_rows = int(
        connection.execute("SELECT COALESCE(SUM(group_size), 0) FROM question_groups").fetchone()[0]
    )
    indexed_variant_rows = int(
        connection.execute("SELECT COALESCE(SUM(response_frequency), 0) FROM response_variants").fetchone()[0]
    )
    if indexed_group_rows != processed or indexed_variant_rows != processed:
        raise RuntimeError("Question/response fingerprint counts do not reconcile to PASS rows")
    return {
        "pass_records_examined": processed,
        "per_shard_rows": dict(per_shard_rows),
        "language_prediction_counts": {
            field: dict(sorted(counts.items())) for field, counts in language_counts.items()
        },
        "quality_flag_counts_overlapping": dict(sorted(quality_flag_counts.items())),
    }


def _create_analysis_table(connection: sqlite3.Connection, seed: int) -> dict[str, Any]:
    group_count = int(connection.execute("SELECT COUNT(*) FROM question_groups").fetchone()[0])
    response_count_distribution = {
        str(int(distinct_count)): int(count)
        for distinct_count, count in connection.execute(
            """SELECT response_count, COUNT(*) FROM (
                    SELECT group_key, COUNT(*) AS response_count
                    FROM response_variants GROUP BY group_key
                ) GROUP BY response_count ORDER BY response_count"""
        )
    }
    multiple_count = int(
        connection.execute("SELECT COUNT(*) FROM multiple_groups").fetchone()[0]
    )
    multi_record_count = int(
        connection.execute("SELECT COALESCE(SUM(group_size), 0) FROM multiple_groups").fetchone()[0]
    )
    variant_total = int(
        connection.execute("SELECT COALESCE(SUM(distinct_response_count), 0) FROM multiple_groups").fetchone()[0]
    )
    connection.execute("CREATE INDEX multiple_groups_rank_idx ON multiple_groups(random_rank)")
    connection.commit()
    return {
        "distinct_question_groups": group_count,
        "multiple_response_group_count": multiple_count,
        "records_in_multiple_response_groups": multi_record_count,
        "response_variants_in_multiple_response_groups": variant_total,
        "distinct_response_count_distribution": response_count_distribution,
    }


def _collect_group_provenance(
    parts: Sequence[Path],
    connection: sqlite3.Connection,
    *,
    batch_size: int,
) -> dict[str, Any]:
    processed = 0
    matching = 0
    multi_language_counts: dict[str, Counter[str]] = {
        "question": Counter(),
        "response": Counter(),
    }
    multi_quality_flags: Counter[str] = Counter()
    next_progress_at = 250_000
    for _, batch in _iter_pass_batches(parts, batch_size):
        values = batch.to_pylist()
        group_keys = sorted({_record_key_values(record)[0] for record in values})
        matching_keys: set[str] = set()
        for offset in range(0, len(group_keys), 500):
            chunk = group_keys[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            matching_keys.update(
                row[0]
                for row in connection.execute(
                    f"SELECT group_key FROM multiple_groups WHERE group_key IN ({placeholders})",
                    chunk,
                )
            )
        records_to_insert: list[tuple[Any, ...]] = []
        for record in values:
            processed += 1
            group_key, _, response_key, _ = _record_key_values(record)
            if group_key not in matching_keys:
                continue
            matching += 1
            row_id = _as_text(record.get("source_row_id"), field_name="source_row_id")
            shard = _as_text(record.get("source_shard"), field_name="source_shard")
            row_number = record.get("source_row_number_1_based")
            if not isinstance(row_number, int):
                raise ValueError(f"Invalid source row number for {row_id}")
            records_to_insert.append(
                (
                    group_key,
                    response_key,
                    row_id,
                    shard,
                    row_number,
                    record.get("dataset_revision"),
                    _as_text(record.get("question"), field_name="question"),
                    _as_text(record.get("response"), field_name="response"),
                    json.dumps(record.get("quality_flags") or [], ensure_ascii=False, separators=(",", ":")),
                    record.get("reason_code"),
                    json.dumps(record.get("secondary_reason_codes") or [], ensure_ascii=False, separators=(",", ":")),
                    record.get("record_fingerprint"),
                    record.get("context_sha256"),
                    record.get("question_language"),
                    record.get("question_language_normalized"),
                    record.get("question_prediction_status"),
                    record.get("question_classifier_score"),
                    record.get("response_language"),
                    record.get("response_language_normalized"),
                    record.get("response_prediction_status"),
                    record.get("response_classifier_score"),
                )
            )
            for field in ("question", "response"):
                multi_language_counts[field][_language_label(record, field)] += 1
            for flag in record.get("quality_flags") or []:
                multi_quality_flags[str(flag)] += 1
        if records_to_insert:
            connection.executemany(
                """INSERT INTO group_records VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                records_to_insert,
            )
            connection.commit()
        if processed >= next_progress_at:
            print(f"Collected provenance for {matching:,} rows in multiple-response groups; scanned {processed:,} PASS rows", flush=True)
            next_progress_at = ((processed // 250_000) + 1) * 250_000
    if processed != sum(pq.ParquetFile(path).metadata.num_rows for path in parts):
        raise RuntimeError("Second PASS scan did not reconcile to PASS Parquet metadata")
    expected_multi_rows = int(
        connection.execute("SELECT COUNT(*) FROM group_records").fetchone()[0]
    )
    if matching != expected_multi_rows:
        raise RuntimeError("Multiple-response provenance rows do not reconcile")
    connection.execute(
        """UPDATE multiple_groups SET has_hindi_or_mixed=1 WHERE group_key IN (
            SELECT group_key FROM group_records
            WHERE question_language_normalized IN ('hi', 'mixed')
               OR response_language_normalized IN ('hi', 'mixed')
               OR lower(COALESCE(question_language, ''))='mixed'
               OR lower(COALESCE(response_language, ''))='mixed'
        )"""
    )
    connection.commit()
    return {
        "pass_records_revisited": processed,
        "records_in_multiple_response_groups": matching,
        "language_prediction_counts": {
            field: dict(sorted(counts.items())) for field, counts in multi_language_counts.items()
        },
        "quality_flag_counts_overlapping": dict(sorted(multi_quality_flags.items())),
    }


def select_review_groups(
    connection: sqlite3.Connection,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a fixed-size priority sample plus a random-ranked remainder."""
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    total_groups = int(connection.execute("SELECT COUNT(*) FROM multiple_groups").fetchone()[0])
    target_size = min(sample_size, total_groups)
    if target_size == 0:
        return [], {"targeted_quota_per_category": 0, "random_remaining_eligible": 0, "random_remaining_selected": 0}
    targeted_quota = math.ceil(target_size / 5)
    selected: dict[str, dict[str, Any]] = {}
    query_templates = [
        (
            "many_distinct_responses",
            "SELECT group_key, group_size, distinct_response_count, min_response_characters, max_response_characters, distinct_surface_forms, has_hindi_or_mixed FROM multiple_groups ORDER BY distinct_response_count DESC, random_rank ASC",
        ),
        (
            "substantial_response_length_spread",
            "SELECT group_key, group_size, distinct_response_count, min_response_characters, max_response_characters, distinct_surface_forms, has_hindi_or_mixed FROM multiple_groups WHERE max_response_characters-min_response_characters >= 100 ORDER BY max_response_characters-min_response_characters DESC, random_rank ASC",
        ),
        (
            "textual_content_difference_candidate",
            "SELECT group_key, group_size, distinct_response_count, min_response_characters, max_response_characters, distinct_surface_forms, has_hindi_or_mixed FROM multiple_groups WHERE distinct_surface_forms > 1 ORDER BY distinct_surface_forms DESC, random_rank ASC",
        ),
        (
            "hindi_or_mixed_language_prediction",
            "SELECT group_key, group_size, distinct_response_count, min_response_characters, max_response_characters, distinct_surface_forms, has_hindi_or_mixed FROM multiple_groups WHERE has_hindi_or_mixed=1 ORDER BY random_rank ASC",
        ),
    ]
    for selection_type, query in query_templates:
        category_added = 0
        for row in connection.execute(query):
            key = row[0]
            if key in selected:
                continue
            selected[key] = _selected_group_row(row, selection_type, None)
            category_added += 1
            if category_added >= targeted_quota or len(selected) >= target_size:
                break
        if len(selected) >= target_size:
            break

    random_remaining_eligible = total_groups - len(selected)
    random_remaining_target = target_size - len(selected)
    random_probability = (
        random_remaining_target / random_remaining_eligible
        if random_remaining_eligible > 0 and random_remaining_target > 0
        else None
    )
    if random_remaining_target:
        if selected:
            keys = list(selected)
            placeholders = ",".join("?" for _ in keys)
            query = (
                "SELECT group_key, group_size, distinct_response_count, min_response_characters, "
                "max_response_characters, distinct_surface_forms, has_hindi_or_mixed "
                f"FROM multiple_groups WHERE group_key NOT IN ({placeholders}) ORDER BY random_rank ASC"
            )
            cursor = connection.execute(query, keys)
        else:
            cursor = connection.execute(
                "SELECT group_key, group_size, distinct_response_count, min_response_characters, max_response_characters, distinct_surface_forms, has_hindi_or_mixed FROM multiple_groups ORDER BY random_rank ASC"
            )
        for row in cursor:
            selected[row[0]] = _selected_group_row(row, "random_remaining", random_probability)
            if len(selected) >= target_size:
                break

    rows = sorted(selected.values(), key=lambda item: (item["selection_type"], item["group_id"]))
    counts = Counter(item["selection_type"] for item in rows)
    return rows, {
        "targeted_quota_per_category": targeted_quota,
        "random_remaining_eligible": random_remaining_eligible,
        "random_remaining_selected": random_remaining_target,
        "random_remaining_inclusion_probability": random_probability,
        "selection_counts_by_type": dict(sorted(counts.items())),
        "targeted_categories_are_probability_samples": False,
        "random_remaining_method": "lowest fixed-seed SHA-256 ranks among groups not already selected",
    }


def _selected_group_row(row: Sequence[Any], selection_type: str, probability: float | None) -> dict[str, Any]:
    key, group_size, variant_count, min_chars, max_chars, surface_count, has_hindi = row
    return {
        "group_key": key,
        "group_id": f"qgrp_{key}",
        "group_size": int(group_size),
        "distinct_response_count": int(variant_count),
        "min_response_characters": int(min_chars),
        "max_response_characters": int(max_chars),
        "response_length_spread": int(max_chars) - int(min_chars),
        "distinct_surface_forms": int(surface_count),
        "has_hindi_or_mixed_prediction": bool(has_hindi),
        "selection_type": selection_type,
        "inclusion_probability": probability,
    }


def _raw_label_counts(connection: sqlite3.Connection, group_key: str, field: str) -> dict[str, int]:
    column = "question_language" if field == "question" else "response_language"
    return {
        str(label if label else "<missing>"): int(count)
        for label, count in connection.execute(
            f"SELECT {column}, COUNT(*) FROM group_records WHERE group_key=? GROUP BY {column} ORDER BY {column}",
            (group_key,),
        )
    }


def _write_json_value(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _write_group_jsonl(connection: sqlite3.Connection, output_path: Path) -> int:
    group_count = 0
    with output_path.open("wb") as raw_output:
        import gzip
        import io

        with gzip.GzipFile(filename="", fileobj=raw_output, mode="wb", compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output:
                groups = connection.execute(
                    """SELECT g.group_key, g.normalized_question, g.question_text,
                              g.group_size, m.distinct_response_count
                       FROM multiple_groups m JOIN question_groups g USING (group_key)
                       ORDER BY m.group_key"""
                )
                for group_key, normalized_question, question_text, group_size, response_count in groups:
                    output.write("{")
                    header = {
                        "group_id": f"qgrp_{group_key}",
                        "normalized_question_key": group_key,
                        "normalized_question": normalized_question,
                        "question_text": question_text,
                        "group_size": int(group_size),
                        "distinct_response_count": int(response_count),
                        "question_language_prediction_counts": _raw_label_counts(connection, group_key, "question"),
                        "response_variants": None,
                    }
                    # Emit the fixed group header and stream nested response provenance.
                    for key, value in header.items():
                        if key == "response_variants":
                            output.write(',"response_variants":[')
                            break
                        if key != "group_id":
                            output.write(",")
                        _write_json_value(output, key)
                        output.write(":")
                        _write_json_value(output, value)
                    variant_cursor = connection.execute(
                        """SELECT response_key, normalized_response, representative_response_text,
                                  normalized_response_characters, response_frequency
                           FROM response_variants WHERE group_key=? ORDER BY response_key""",
                        (group_key,),
                    )
                    for variant_index, variant in enumerate(variant_cursor):
                        if variant_index:
                            output.write(",")
                        response_key, normalized_response, representative, char_count, frequency = variant
                        output.write("{")
                        variant_header = {
                            "response_variant_id": response_key,
                            "normalized_response": normalized_response,
                            "representative_response_text": representative,
                            "normalized_response_characters": int(char_count),
                            "response_frequency": int(frequency),
                            "response_language_prediction_counts": {
                                str(label if label else "<missing>"): int(count)
                                for label, count in connection.execute(
                                    "SELECT response_language, COUNT(*) FROM group_records WHERE group_key=? AND response_key=? GROUP BY response_language ORDER BY response_language",
                                    (group_key, response_key),
                                )
                            },
                            "source_records": None,
                        }
                        for field_index, (key, value) in enumerate(variant_header.items()):
                            if key == "source_records":
                                output.write(',"source_records":[')
                                break
                            if field_index:
                                output.write(",")
                            _write_json_value(output, key)
                            output.write(":")
                            _write_json_value(output, value)
                        record_cursor = connection.execute(
                            """SELECT source_row_id, dataset_revision, source_shard,
                                      source_row_number_1_based, question_text, response_text,
                                      quality_flags_json, reason_code, secondary_reason_codes_json,
                                      record_fingerprint, context_sha256, question_language,
                                      question_language_normalized, question_prediction_status,
                                      question_classifier_score, response_language,
                                      response_language_normalized, response_prediction_status,
                                      response_classifier_score
                               FROM group_records WHERE group_key=? AND response_key=?
                               ORDER BY source_shard, source_row_number_1_based""",
                            (group_key, response_key),
                        )
                        for record_index, record in enumerate(record_cursor):
                            if record_index:
                                output.write(",")
                            record_fields = (
                                "source_row_id", "dataset_revision", "source_shard",
                                "source_row_number_1_based", "question_text", "response_text",
                                "quality_flags_json", "reason_code", "secondary_reason_codes_json",
                                "record_fingerprint", "context_sha256", "question_language",
                                "question_language_normalized", "question_prediction_status",
                                "question_classifier_score", "response_language",
                                "response_language_normalized", "response_prediction_status",
                                "response_classifier_score",
                            )
                            provenance = dict(zip(record_fields, record, strict=True))
                            provenance["quality_flags"] = json.loads(provenance.pop("quality_flags_json"))
                            provenance["secondary_reason_codes"] = json.loads(
                                provenance.pop("secondary_reason_codes_json")
                            )
                            _write_json_value(output, provenance)
                        output.write("]}")
                    output.write("]}\n")
                    group_count += 1
                    if group_count % 1_000 == 0:
                        print(f"Wrote {group_count:,} multiple-response groups", flush=True)
    return group_count


def _group_for_csv(connection: sqlite3.Connection, group_key: str) -> dict[str, Any]:
    group = connection.execute(
        """SELECT g.normalized_question, g.question_text, g.group_size,
                  m.distinct_response_count
           FROM question_groups g JOIN multiple_groups m USING (group_key)
           WHERE g.group_key=?""",
        (group_key,),
    ).fetchone()
    if not group:
        raise KeyError(group_key)
    normalized_question, question_text, group_size, variant_count = group
    variants: list[dict[str, Any]] = []
    for variant_index, row in enumerate(
        connection.execute(
            """SELECT response_key, normalized_response, representative_response_text,
                      normalized_response_characters, response_frequency
               FROM response_variants WHERE group_key=? ORDER BY response_key""",
            (group_key,),
        ),
        start=1,
    ):
        response_key, normalized_response, representative, char_count, frequency = row
        lang_counts = {
            str(label if label else "<missing>"): int(count)
            for label, count in connection.execute(
                "SELECT response_language, COUNT(*) FROM group_records WHERE group_key=? AND response_key=? GROUP BY response_language ORDER BY response_language",
                (group_key, response_key),
            )
        }
        sample_records = [
            {
                "source_row_id": source_row_id,
                "source_shard": shard,
                "source_row_number_1_based": int(row_number),
                "question_language": q_lang,
                "response_language": r_lang,
            }
            for source_row_id, shard, row_number, q_lang, r_lang in connection.execute(
                """SELECT source_row_id, source_shard, source_row_number_1_based,
                          question_language, response_language
                   FROM group_records WHERE group_key=? AND response_key=?
                   ORDER BY source_shard, source_row_number_1_based LIMIT 5""",
                (group_key, response_key),
            )
        ]
        variants.append(
            {
                "variant_number": variant_index,
                "response_variant_id": response_key,
                "response_frequency": int(frequency),
                "normalized_response_characters": int(char_count),
                "normalized_response": normalized_response,
                "representative_response_text": representative,
                "response_language_prediction_counts": lang_counts,
                "representative_source_records": sample_records,
            }
        )
    provenance_rows = [
        {
            "source_row_id": source_row_id,
            "source_shard": source_shard,
            "source_row_number_1_based": int(row_number),
            "question_language": question_language,
            "response_language": response_language,
            "quality_flags": json.loads(flags_json),
        }
        for source_row_id, source_shard, row_number, question_language, response_language, flags_json in connection.execute(
            """SELECT source_row_id, source_shard, source_row_number_1_based,
                      question_language, response_language, quality_flags_json
               FROM group_records WHERE group_key=?
               ORDER BY source_shard, source_row_number_1_based LIMIT 20""",
            (group_key,),
        )
    ]
    return {
        "normalized_question": normalized_question,
        "question_text": question_text,
        "group_size": int(group_size),
        "distinct_response_count": int(variant_count),
        "response_variants": variants,
        "representative_source_records": provenance_rows,
    }


def _write_review_csv(
    connection: sqlite3.Connection,
    selected: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    seed: int,
    preview_limit: int = 30_000,
) -> None:
    columns = [
        "group_id",
        "sampling_selection_type",
        "sampling_seed",
        "inclusion_probability",
        "group_size",
        "distinct_response_count",
        "normalized_question_key",
        "question_text",
        "response_variants_json",
        "response_variants_truncated_for_spreadsheet",
        "representative_source_records_json",
        "machine_audit_lookup",
        "review_disposition",
        "review_notes",
        "reviewer_id",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in selected:
            group = _group_for_csv(connection, str(item["group_key"]))
            variants_json = json.dumps(group["response_variants"], ensure_ascii=False, separators=(",", ":"))
            truncated = len(variants_json) > preview_limit
            if truncated:
                preview_variants = []
                current_size = 2
                for variant in group["response_variants"]:
                    clone = dict(variant)
                    for text_key in ("normalized_response", "representative_response_text"):
                        text = str(clone.get(text_key, ""))
                        if len(text) > 1_500:
                            clone[text_key] = text[:1_500] + " [preview truncated; full text in machine audit]"
                    encoded = json.dumps(clone, ensure_ascii=False, separators=(",", ":"))
                    if current_size + len(encoded) > preview_limit:
                        break
                    preview_variants.append(clone)
                    current_size += len(encoded) + 1
                variants_json = json.dumps(preview_variants, ensure_ascii=False, separators=(",", ":"))
            writer.writerow(
                {
                    "group_id": item["group_id"],
                    "sampling_selection_type": item["selection_type"],
                    "sampling_seed": seed,
                    "inclusion_probability": item.get("inclusion_probability"),
                    "group_size": item["group_size"],
                    "distinct_response_count": item["distinct_response_count"],
                    "normalized_question_key": item["group_key"],
                    "question_text": group["question_text"],
                    "response_variants_json": variants_json,
                    "response_variants_truncated_for_spreadsheet": str(truncated).lower(),
                    "representative_source_records_json": json.dumps(
                        group["representative_source_records"], ensure_ascii=False, separators=(",", ":")
                    ),
                    "machine_audit_lookup": f"multiple_response_groups.jsonl.gz#group_id={item['group_id']}",
                    "review_disposition": "",
                    "review_notes": "",
                    "reviewer_id": "",
                }
            )


def _format_group_example(connection: sqlite3.Connection, group_key: str) -> dict[str, Any]:
    group = _group_for_csv(connection, group_key)
    return {
        "group_id": f"qgrp_{group_key}",
        "question_text": group["question_text"],
        "group_size": group["group_size"],
        "distinct_response_count": group["distinct_response_count"],
        "response_variants": [
            {
                "frequency": item["response_frequency"],
                "response": item["representative_response_text"][:500],
            }
            for item in group["response_variants"][:3]
        ],
    }


def _write_report(
    path: Path,
    *,
    summary: Mapping[str, Any],
    sampling: Mapping[str, Any],
    examples: Mapping[str, Any],
) -> None:
    distribution = summary["distinct_response_count_distribution"]
    lines = [
        "# Multiple responses per question: Stage 3 PASS audit",
        "",
        f"- Dataset revision: `{summary['dataset_revision']}`",
        f"- Stage 3 PASS records examined: **{summary['pass_records_examined']:,}**",
        f"- Distinct normalized question groups: **{summary['distinct_question_groups']:,}**",
        f"- Groups with multiple normalized responses: **{summary['multiple_response_group_count']:,}**",
        f"- Records in those groups: **{summary['records_in_multiple_response_groups']:,}**",
        f"- Distinct response variants across those groups: **{summary['response_variants_in_multiple_response_groups']:,}**",
        f"- All findings describe Stage 3 PASS only; this run did not reread or modify Stage 3 outputs or raw shards.",
        "",
        "A group with multiple answers is a review signal, not proof that its responses conflict or are legally wrong.",
        "",
        "## Distinct response count per question group",
        "",
        "| Distinct normalized responses | Question groups |",
        "|---:|---:|",
    ]
    for response_count, question_count in sorted(distribution.items(), key=lambda row: int(row[0])):
        lines.append(f"| {int(response_count):,} | {int(question_count):,} |")
    lines.extend(["", "## Classifier observations", ""])
    for label, title in (("pass_language_prediction_counts", "All PASS records"), ("multiple_group_language_prediction_counts", "Records in multiple-response groups")):
        lines.append(f"### {title}")
        lines.append("")
        lines.append("These are raw classifier labels, not human-verified language counts.")
        for field, counts in summary[label].items():
            lines.append(f"- **{field}:** `{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`")
        lines.append("")
    lines.extend(
        [
            "## Review sample",
            "",
            f"- Requested groups: {sampling['requested_sample_size']:,}; selected: {sampling['actual_sample_size']:,}.",
            f"- Fixed seed: `{sampling['seed']}`.",
            f"- Selection counts by type: `{json.dumps(sampling['selection_counts_by_type'], sort_keys=True)}`.",
            "- The first four priority categories are targeted and are not probability samples. The remaining stratum uses deterministic SHA-256 ranking; its inclusion probability is reported in the CSV.",
            "- Selection heuristics rank groups with many responses, large response-length spread, differing case/punctuation-insensitive text signatures, and Hindi/mixed classifier observations. These heuristics do not label answer meaning or conflict.",
            "- Complete the blank review fields using the labels described in `review_instructions.md`: `VALID_ALTERNATIVE_ANSWERS`, `POTENTIAL_CONFLICT`, `DUPLICATE_VARIANT`, or `UNCERTAIN`.",
            "",
            "## Representative response differences",
            "",
            "### Case/punctuation-insensitive surface-screen matches",
        ]
    )
    if examples.get("surface_variant_example"):
        lines.append("These examples map to one screening signature after lowercasing and removing punctuation/spacing, while retaining separate NFC/whitespace response variants. This screen is for sampling and is not evidence of semantic equivalence.")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(examples["surface_variant_example"], ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("No example was found in this PASS subset.")
    lines.extend(["", "### Textually different response candidates", ""])
    if examples.get("text_difference_example"):
        lines.append("These examples differ beyond the case/punctuation-insensitive screen. That does not establish that they are semantically or legally contradictory.")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(examples["text_difference_example"], ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("No example was found in this PASS subset.")
    lines.extend(
        [
            "",
            "## Method and limitations",
            "",
            "- Question grouping and response variants reuse the audit's NFC Unicode normalization plus whitespace collapse and `stable_hash([field])`. Case, punctuation, names, digits, and legal citations are otherwise preserved.",
            "- Original question and response text is retained in the complete compressed JSONL, with all source row IDs, shard/row provenance, classifier labels/scores, quality flags, and record/context fingerprints for every member record.",
            "- The full machine audit contains every multiple-response group. The CSV has one row per selected group, blank human-review fields, and a group-ID lookup into the complete artifact. Large response bundles may be preview-truncated in CSV; the complete texts remain in JSONL.",
            "- Only the Stage 3 PASS pool was examined. QUARANTINE and REJECT groups are outside this count. No contexts were read or included; Stage 3's context fingerprint is retained as provenance only.",
            "- Counts of distinct responses are normalized-string counts, not semantic answer equivalence or legal-conflict adjudication. A prior audit count must not be assumed to apply to this PASS subset.",
            "- Sampling heuristics and classifier labels can miss substantive differences or prioritize harmless differences. Human review is required before any group-level training decision.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_instructions(path: Path, machine_artifact_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Multiple-response review instructions",
                "",
                "Review each sampled question group as one unit. The response variants and their frequencies are shown in the CSV; use `group_id` to find the complete record-level provenance in the compressed JSONL audit.",
                "",
                "Do not use source context to infer what the question or answer should mean. The package contains classifier observations and source pointers for provenance; it does not establish legal correctness.",
                "",
                "## Review dispositions",
                "",
                "- `VALID_ALTERNATIVE_ANSWERS`: responses can reasonably answer the same question without a material conflict.",
                "- `POTENTIAL_CONFLICT`: responses appear to give materially incompatible answers for the same normalized question and need adjudication.",
                "- `DUPLICATE_VARIANT`: responses appear equivalent and differ only in formatting or wording that does not change the answer.",
                "- `UNCERTAIN`: evidence is insufficient to choose another disposition.",
                "",
                "These are reviewer labels, not automatic classifications. Do not label a legal contradiction without appropriate legal/source review.",
                "",
                "Fill `review_disposition`, `review_notes`, and `reviewer_id` manually. Leave them blank until a human review is performed. Do not overwrite classifier predictions or source text.",
                "",
                f"Complete group records: `{machine_artifact_name}`. The CSV may preview-truncate unusually large response bundles; the machine artifact retains full original text and provenance.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_multiple_response_audit(
    *,
    stage3_dir: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    stage3_dir = stage3_dir.resolve()
    output_dir = output_dir.resolve()
    if stage3_dir == output_dir or stage3_dir in output_dir.parents or output_dir in stage3_dir.parents:
        raise ValueError("Audit output must be separate from the Stage 3 output tree")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")

    stage3_summary, _, _ = _stage3_input_manifest(stage3_dir)
    expected_pass = int(stage3_summary["outcome_counts"]["PASS"])
    pass_dir = stage3_dir / "outcome_partitions" / "PASS"
    parts, pass_part_counts = _pass_parts(pass_dir, expected_pass)
    available_bytes = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else Path.cwd()).free
    pass_bytes = sum(path.stat().st_size for path in parts)
    minimum_free = max(4 * 1024**3, 2 * pass_bytes + 1 * 1024**3)
    if available_bytes < minimum_free:
        raise RuntimeError(
            f"Insufficient free disk space: {available_bytes:,} available; "
            f"estimated minimum is {minimum_free:,} bytes"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_parent = output_dir / ".work"
    work_parent.mkdir()
    work_dir = Path(tempfile.mkdtemp(prefix="multiple_response_", dir=work_parent))
    db_path = work_dir / "group_index.sqlite3"
    try:
        connection = _open_work_index(db_path)
        first_pass = _collect_first_pass(parts, connection, batch_size=batch_size)
        if first_pass["pass_records_examined"] != expected_pass:
            raise RuntimeError(
                f"Read {first_pass['pass_records_examined']:,} PASS rows; Stage 3 summary says {expected_pass:,}"
            )
        _create_multiple_group_tables(connection, seed)
        aggregate = _create_analysis_table(connection, seed)
        second_pass = _collect_group_provenance(parts, connection, batch_size=batch_size)
        selected, sampling_selection = select_review_groups(connection, sample_size=sample_size)

        group_jsonl = output_dir / "multiple_response_groups.jsonl.gz"
        written_groups = _write_group_jsonl(connection, group_jsonl)
        if written_groups != aggregate["multiple_response_group_count"]:
            raise RuntimeError(
                f"Wrote {written_groups:,} groups; expected {aggregate['multiple_response_group_count']:,}"
            )
        review_csv = output_dir / "multiple_response_review_sample.csv"
        _write_review_csv(connection, selected, review_csv, seed=seed)
        instructions_path = output_dir / "review_instructions.md"
        _write_review_instructions(instructions_path, group_jsonl.name)

        surface_example_row = connection.execute(
            """SELECT group_key FROM multiple_groups
               WHERE distinct_response_count=2 AND distinct_surface_forms=1
               ORDER BY random_rank LIMIT 1"""
        ).fetchone()
        text_difference_row = connection.execute(
            """SELECT group_key FROM multiple_groups
               WHERE distinct_surface_forms > 1
               ORDER BY distinct_surface_forms DESC, random_rank LIMIT 1"""
        ).fetchone()
        examples = {
            "surface_variant_example": _format_group_example(connection, surface_example_row[0])
            if surface_example_row
            else None,
            "text_difference_example": _format_group_example(connection, text_difference_row[0])
            if text_difference_row
            else None,
        }
        summary: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_revision": stage3_summary["dataset_revision"],
            "stage3_input_directory": str(stage3_dir),
            "pass_partition_directory": str(pass_dir),
            "output_directory": str(output_dir),
            "normalization": "existing audit normalization: Unicode NFC and whitespace collapse; no case or punctuation normalization",
            "group_id_method": "qgrp_ + existing stable_hash([original question])",
            "response_variant_id_method": "existing stable_hash([original response])",
            "pass_records_examined": first_pass["pass_records_examined"],
            "pass_parquet_part_count": len(parts),
            "pass_parquet_part_row_counts": pass_part_counts,
            **aggregate,
            "multiple_group_provenance_record_count": second_pass["records_in_multiple_response_groups"],
            "pass_language_prediction_counts": first_pass["language_prediction_counts"],
            "multiple_group_language_prediction_counts": second_pass["language_prediction_counts"],
            "pass_quality_flag_counts_overlapping": first_pass["quality_flag_counts_overlapping"],
            "multiple_group_quality_flag_counts_overlapping": second_pass["quality_flag_counts_overlapping"],
            "sampling": {
                "requested_sample_size": sample_size,
                "actual_sample_size": len(selected),
                "seed": seed,
                **sampling_selection,
                "selected_group_ids": [item["group_id"] for item in selected],
                "method_is_representative_probability_sample": False,
            },
            "sample_selection_metadata": selected,
            "representative_examples": examples,
            "output_files": {
                "complete_group_audit": group_jsonl.name,
                "human_review_csv": review_csv.name,
                "review_instructions": instructions_path.name,
                "summary": "multiple_response_summary.json",
                "report": "multiple_response_report.md",
            },
            "memory_model": {
                "pass_read_batch_size": batch_size,
                "full_pass_rows_loaded_into_python_memory": False,
                "sqlite_work_index_disk_backed": True,
                "sqlite_cache_size_bytes": 32 * 1024 * 1024,
                "provenance_loaded_into_memory": "one Arrow batch and one group/response variant at a time, except bounded review sample",
            },
            "limitations": [
                "Only Stage 3 PASS records are included; QUARANTINE and REJECT records are excluded.",
                "Response variants are defined by NFC and whitespace normalization, not semantic similarity.",
                "Case/punctuation-insensitive surface signatures are used only to prioritize examples and never alter response grouping.",
                "Classifier labels and scores are observations, not human-confirmed language or answer truth.",
                "Source contexts were not read; source row IDs, fingerprints, and quality flags are retained for provenance.",
                "The audit does not determine whether responses are compatible, alternative, or legally contradictory.",
            ],
        }
        summary_path = output_dir / "multiple_response_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path = output_dir / "multiple_response_report.md"
        _write_report(report_path, summary=summary, sampling=summary["sampling"], examples=examples)
        connection.close()
        return summary
    finally:
        if "connection" in locals():
            try:
                connection.close()
            except sqlite3.Error:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            work_parent.rmdir()
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.batch_size < 1 or args.sample_size < 0:
        parser.error("batch size must be positive and sample size non-negative")
    return args


def main() -> None:
    args = parse_args()
    summary = run_multiple_response_audit(
        stage3_dir=args.stage3_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(
        f"PASS rows {summary['pass_records_examined']:,}; "
        f"question groups {summary['distinct_question_groups']:,}; "
        f"multiple-response groups {summary['multiple_response_group_count']:,}; "
        f"review sample {summary['sampling']['actual_sample_size']:,}; "
        f"wrote {summary['output_directory']}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
