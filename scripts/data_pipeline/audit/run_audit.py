"""Read-only, batch-based audit for the LegalBrain AI Parquet dataset.

This script reports data-quality signals; it does not modify or reject source rows.
Run from the project root, for example:
  uv run python scripts/data_pipeline/audit/run_audit.py --max-rows-per-shard 1000
  uv run python scripts/data_pipeline/audit/run_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow
import pyarrow.parquet as pq

try:
    import langid  # Optional: used only to profile a sample of rows.
except ImportError:  # pragma: no cover - depends on optional local package
    langid = None

REQUIRED_COLUMNS = ("context", "question", "response")

# These are review flags, not automatic rejection criteria.
SHORT_QUESTION_CHARS = 8
SHORT_RESPONSE_CHARS = 20
LONG_QUESTION_CHARS = 2_000
LONG_RESPONSE_CHARS = 10_000
LONG_CONTEXT_CHARS = 50_000
PREVIEW_CHARS = 2_000
CONTEXT_PREVIEW_CHARS = 3_000

CONTEXT_DEPENDENT_QUESTION = re.compile(
    r"\b(?:in this case|in the present case|what did the court|"
    r"as mentioned above|as stated above|according to the passage|"
    r"the impugned order|the aforesaid|the above-mentioned)\b|"
    r"\b(?:this|the above|above-mentioned|aforementioned|following|said)\s+"
    r"(?:case|judgment|order|passage|provision|article|section|document|"
    r"paragraph|petitioner|respondent|court)\b",
    re.IGNORECASE,
)

# Form-feed (\x0c) commonly represents a PDF page break in extracted legal text.
# Track it separately rather than treating every page break as corruption.
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
FORM_FEED = "\x0c"

PLACEHOLDER_RESPONSES = {
    "answer is not available",
    "the answer is not available",
    "answer not available",
    "the answer not available",
    "not available",
    "no answer available",
    "answer unavailable",
    "the answer is unavailable",
}

PLACEHOLDER_NORMALIZER = re.compile(r"[^a-z0-9]+", re.IGNORECASE)

# Only flag clear declarative clauses with a recognizable legal-case subject and
# finite predicate. Many valid prompts are fragments and do not end in '?'.
QUESTION_OR_TASK_LEAD = re.compile(
    r"^\s*(?:(?:please\s+)?(?:what|who|whom|whose|which|when|where|why|how|"
    r"whether|is|are|was|were|do|does|did|can|could|would|should|will|shall|"
    r"may|might|must|has|have|had|explain|describe|discuss|outline|summari[sz]e|"
    r"identify|list|state|define|clarify|compare|contrast|analyse|analyze|"
    r"evaluate|determine|provide|give|tell|elaborate|illustrate|specify|mention)\b)",
    re.IGNORECASE,
)
STATEMENT_LIKE_QUESTION = re.compile(
    r"^\s*(?:"
    r"(?:the|this|that)\s+(?:(?:trial|appellate|supreme)\s+)?"
    r"(?:court|judge|section|article|rule|provision|case|appeal|"
    r"petitioner|respondent|appellant|plaintiff|defendant|order|judgment)"
    r"|it|they|he|she|we|i|the parties"
    r")\s+(?:(?:clearly|correctly|wrongly|properly|ultimately|"
    r"therefore|accordingly|also|not)\s+){0,2}"
    r"(?:is|are|was|were|has|have|had|does|do|did|can|could|would|"
    r"should|will|shall|may|might|must|holds?|held|finds?|found|"
    r"determines?|determined|concludes?|concluded|observes?|observed|"
    r"states?|stated|provides?|provided|requires?|required|"
    r"establishes?|established|applies|applied|dismisses|dismissed|"
    r"allows|allowed|prohibits|prohibited|protects|protected)\b",
    re.IGNORECASE,
)

# Flag likely citation gaps only in explicit citation frames or where a visible
# blank placeholder appears. Generic phrases such as "which section of the Act"
# and "rule of law" are not evidence that a reference number is missing.
MISSING_LEGAL_REFERENCE_NUMBER = re.compile(
    r"(?:\b(?:under|pursuant\s+to|according\s+to|in\s+accordance\s+with|"
    r"in\s+terms\s+of|as\s+per)\s+"
    r"(?:the\s+)?(?:sections?|articles?|rules?|regulations?|clauses?|"
    r"sub-?sections?)\s+(?:of|under|in)\b"
    r"(?!\s+(?:consideration|discussion|review|question|law)\b))"
    r"|(?:\b(?:sections?|articles?|rules?|regulations?|clauses?|"
    r"sub-?sections?)\s+(?:_{2,}|\[\s*(?:number|missing)\s*\]))",
    re.IGNORECASE,
)

LENGTH_BUCKETS = (
    (0, 0, "0"),
    (1, 20, "1-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, 250, "101-250"),
    (251, 500, "251-500"),
    (501, 1_000, "501-1000"),
    (1_001, 2_000, "1001-2000"),
    (2_001, 4_000, "2001-4000"),
    (4_001, 8_000, "4001-8000"),
    (8_001, 10**18, "8001+"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local Parquet shards without modifying raw data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/indian-legal-supervised-fine-tuning-data/data"),
        help="Directory containing the downloaded Parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports/audit"),
        help="Directory for JSON reports and review samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2_048,
        help="Rows read per Arrow batch (default: 2048).",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=20,
        help="Uniform random review samples retained from each shard.",
    )
    parser.add_argument(
        "--flag-samples-per-category",
        type=int,
        default=20,
        help="Maximum examples retained for each audit flag.",
    )
    parser.add_argument(
        "--language-sample-size",
        type=int,
        default=5_000,
        help="Number of random nontrivial question/answer pairs for optional language profiling.",
    )
    parser.add_argument(
        "--max-rows-per-shard",
        type=int,
        default=0,
        help="Development smoke test only. 0 means scan every row; otherwise cap each shard.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducible sampling."
    )
    parser.add_argument(
        "--dataset-revision",
        default="unrecorded",
        help="Revision/hash of the dataset snapshot that was downloaded.",
    )
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Keep the temporary SQLite fingerprint index in the report directory.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.samples_per_shard < 0 or args.flag_samples_per_category < 0:
        parser.error("sample counts cannot be negative")
    if args.language_sample_size < 0 or args.max_rows_per_shard < 0:
        parser.error("sample/row limits cannot be negative")
    return args


def normalize_for_comparison(value: str | None) -> str:
    """Normalize whitespace and Unicode for fingerprints, not source files."""
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFC", value).split())


def stable_hash(values: list[str | None]) -> str:
    """Hash a structured list without ambiguity between fields."""
    normalized = [
        None if value is None else normalize_for_comparison(value)
        for value in values
    ]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def length_bucket(length: int) -> str:
    for lower, upper, label in LENGTH_BUCKETS:
        if lower <= length <= upper:
            return label
    return "8001+"


def new_field_stats() -> dict[str, Any]:
    return {
        "string_values": 0,
        "null_values": 0,
        "non_string_values": 0,
        "blank_values": 0,
        "total_characters": 0,
        "min_characters": None,
        "max_characters": 0,
        "length_buckets": {label: 0 for _, _, label in LENGTH_BUCKETS},
        "values_with_control_characters": 0,
        "values_with_form_feed_page_break": 0,
        "form_feed_page_break_occurrences": 0,
        "values_with_replacement_character": 0,
    }


def observe_field(stats: dict[str, Any], value: Any) -> None:
    if value is None:
        stats["null_values"] += 1
        return
    if not isinstance(value, str):
        stats["non_string_values"] += 1
        return

    length = len(value)
    stats["string_values"] += 1
    stats["total_characters"] += length
    stats["min_characters"] = (
        length
        if stats["min_characters"] is None
        else min(stats["min_characters"], length)
    )
    stats["max_characters"] = max(stats["max_characters"], length)
    stats["length_buckets"][length_bucket(length)] += 1

    if not value.strip():
        stats["blank_values"] += 1
    if CONTROL_CHARACTERS.search(value):
        stats["values_with_control_characters"] += 1
    if FORM_FEED in value:
        stats["values_with_form_feed_page_break"] += 1
        stats["form_feed_page_break_occurrences"] += value.count(FORM_FEED)
    if "\ufffd" in value:
        stats["values_with_replacement_character"] += 1


def finalize_field_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    count = result["string_values"]
    result["mean_characters"] = (
        round(result["total_characters"] / count, 2) if count else None
    )
    if result["min_characters"] is None:
        result["min_characters"] = None
    return result


def safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def get_record_flags(record: dict[str, Any]) -> list[str]:
    """Return heuristic flags for review; none are automatic rejection rules."""
    flags: list[str] = []
    for field in REQUIRED_COLUMNS:
        value = record.get(field)
        if value is None:
            flags.append(f"missing_{field}")
        elif not isinstance(value, str):
            flags.append(f"non_string_{field}")
        elif not value.strip():
            flags.append(f"blank_{field}")
        if isinstance(value, str) and CONTROL_CHARACTERS.search(value):
            flags.append(f"control_characters_{field}")
        if isinstance(value, str) and "\ufffd" in value:
            flags.append(f"replacement_character_{field}")

    question = safe_text(record.get("question")).strip()
    response = safe_text(record.get("response")).strip()
    context = safe_text(record.get("context"))

    if question and len(question) < SHORT_QUESTION_CHARS:
        flags.append("short_question")
    if response and len(response) < SHORT_RESPONSE_CHARS:
        flags.append("short_response")
    if len(question) > LONG_QUESTION_CHARS:
        flags.append("long_question")
    if len(response) > LONG_RESPONSE_CHARS:
        flags.append("long_response")
    if len(context) > LONG_CONTEXT_CHARS:
        flags.append("long_context")
    if question and CONTEXT_DEPENDENT_QUESTION.search(question):
        flags.append("possible_context_dependent_question")

    if response:
        placeholder_normalized = PLACEHOLDER_NORMALIZER.sub(" ", response.lower()).strip()
        if placeholder_normalized in PLACEHOLDER_RESPONSES:
            flags.append("placeholder_response")

    if (
        question
        and "?" not in question
        and not QUESTION_OR_TASK_LEAD.search(question)
        and STATEMENT_LIKE_QUESTION.search(question)
    ):
        flags.append("statement_like_question")

    if any(
        isinstance(record.get(field), str)
        and MISSING_LEGAL_REFERENCE_NUMBER.search(record[field])
        for field in REQUIRED_COLUMNS
    ):
        flags.append("missing_legal_reference_number")

    return sorted(set(flags))


def sample_preview(record: dict[str, Any], shard: str, row_number: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_shard": shard,
        "source_row_number_1_based": row_number,
        "field_lengths_chars": {},
        "preview_truncated": {},
    }
    for field in REQUIRED_COLUMNS:
        value = record.get(field)
        text = value if isinstance(value, str) else "" if value is None else str(value)
        limit = CONTEXT_PREVIEW_CHARS if field == "context" else PREVIEW_CHARS
        result[field] = text[:limit]
        result["field_lengths_chars"][field] = len(text)
        result["preview_truncated"][field] = len(text) > limit
    result["audit_flags"] = get_record_flags(record)
    return result


def add_reservoir_sample(
    reservoir_state: dict[str, Any],
    item: dict[str, Any],
    capacity: int,
    rng: random.Random,
) -> None:
    if capacity <= 0:
        return
    reservoir_state["seen"] += 1
    seen = reservoir_state["seen"]
    samples: list[dict[str, Any]] = reservoir_state["samples"]
    if len(samples) < capacity:
        samples.append(item)
        return
    replacement_index = rng.randrange(seen)
    if replacement_index < capacity:
        samples[replacement_index] = item


def initialize_index(connection: sqlite3.Connection) -> None:
    # This index is rebuildable; settings favor speed over crash durability.
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE TABLE seen_records (record_hash TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    connection.execute(
        """CREATE TABLE question_answers (
            question_hash TEXT NOT NULL,
            response_hash TEXT NOT NULL,
            occurrences INTEGER NOT NULL,
            PRIMARY KEY (question_hash, response_hash)
        ) WITHOUT ROWID"""
    )
    connection.commit()


def update_fingerprints(
    connection: sqlite3.Connection,
    records: list[dict[str, Any]],
) -> int:
    record_hashes: list[tuple[str]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()

    for record in records:
        record_hashes.append(
            (stable_hash([record.get(field) if isinstance(record.get(field), str) or record.get(field) is None else str(record.get(field)) for field in REQUIRED_COLUMNS]),)
        )

        question = safe_text(record.get("question"))
        response = safe_text(record.get("response"))
        if question.strip() and response.strip():
            pair_counts[(stable_hash([question]), stable_hash([response]))] += 1

    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO seen_records(record_hash) VALUES (?)",
        record_hashes,
    )
    inserted = connection.total_changes - before
    duplicates_in_batch_and_prior = len(record_hashes) - inserted

    connection.executemany(
        """INSERT INTO question_answers(question_hash, response_hash, occurrences)
           VALUES (?, ?, ?)
           ON CONFLICT(question_hash, response_hash)
           DO UPDATE SET occurrences = occurrences + excluded.occurrences""",
        [(question_hash, response_hash, count)
         for (question_hash, response_hash), count in pair_counts.items()],
    )
    connection.commit()
    return duplicates_in_batch_and_prior


def summarize_fingerprints(connection: sqlite3.Connection, total_rows: int) -> dict[str, Any]:
    unique_records = connection.execute(
        "SELECT COUNT(*) FROM seen_records"
    ).fetchone()[0]
    unique_questions = connection.execute(
        "SELECT COUNT(DISTINCT question_hash) FROM question_answers"
    ).fetchone()[0]
    unique_qa_pairs = connection.execute(
        "SELECT COUNT(*) FROM question_answers"
    ).fetchone()[0]
    repeated_qa_extra_occurrences = connection.execute(
        "SELECT COALESCE(SUM(occurrences - 1), 0) FROM question_answers"
    ).fetchone()[0]
    conflicting_questions = connection.execute(
        """SELECT COUNT(*) FROM (
            SELECT question_hash FROM question_answers
            GROUP BY question_hash HAVING COUNT(*) > 1
        )"""
    ).fetchone()[0]

    return {
        "normalized_unique_full_records": unique_records,
        "normalized_duplicate_full_record_occurrences": max(0, total_rows - unique_records),
        "unique_nonblank_questions_with_nonblank_answers": unique_questions,
        "unique_question_response_pairs": unique_qa_pairs,
        "repeated_question_response_pair_extra_occurrences": repeated_qa_extra_occurrences,
        "questions_with_multiple_distinct_responses": conflicting_questions,
        "notes": [
            "Fingerprints normalize Unicode to NFC and collapse whitespace; source text is not altered.",
            "A question with multiple responses is a review signal, not proof that either response is wrong.",
            "Blank questions and blank responses are excluded from question-response conflict metrics.",
        ],
    }


def classify_language_sample(samples: list[dict[str, str]]) -> dict[str, Any]:
    if langid is None:
        return {
            "available": False,
            "sampled_texts": 0,
            "reason": "Optional dependency 'langid' is not installed. Install it with: uv add langid",
        }

    counts: Counter[str] = Counter()
    classified = 0
    skipped_short = 0
    for item in samples:
        text = item["text"].strip()
        if len(text) < 40:
            skipped_short += 1
            continue
        try:
            language, _score = langid.classify(text[:2_000])
            counts[language] += 1
            classified += 1
        except Exception:
            counts["classification_error"] += 1

    return {
        "available": True,
        "method": "langid on a uniform random sample of question + response text; estimates only",
        "sampled_texts_requested": len(samples),
        "classified_texts": classified,
        "skipped_under_40_characters": skipped_short,
        "language_counts": dict(counts.most_common()),
        "warning": "Short legal phrases and mixed-language text can be misclassified; validate a manual sample before filtering languages.",
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def scan_files(args: argparse.Namespace, files: list[Path], connection: sqlite3.Connection) -> dict[str, Any]:
    rng = random.Random(args.seed)
    language_rng = random.Random(args.seed + 1009)

    global_fields = {field: new_field_stats() for field in REQUIRED_COLUMNS}
    global_flags: Counter[str] = Counter()
    total_rows = 0
    duplicate_rows = 0
    shard_summaries: list[dict[str, Any]] = []
    schema_by_shard: dict[str, list[str]] = {}
    random_samples_by_shard: list[dict[str, Any]] = []
    flag_reservoirs: dict[str, dict[str, Any]] = {}
    language_reservoir: dict[str, Any] = {"seen": 0, "samples": []}
    missing_required_schema: dict[str, list[str]] = {}

    for file_number, file_path in enumerate(files, start=1):
        file_size = file_path.stat().st_size
        parquet_file = pq.ParquetFile(file_path)
        schema_names = list(parquet_file.schema_arrow.names)
        schema_by_shard[file_path.name] = schema_names
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(schema_names))
        if missing_columns:
            missing_required_schema[file_path.name] = missing_columns
            raise ValueError(
                f"{file_path} is missing required columns: {', '.join(missing_columns)}"
            )

        metadata_rows = parquet_file.metadata.num_rows
        shard_fields = {field: new_field_stats() for field in REQUIRED_COLUMNS}
        shard_flags: Counter[str] = Counter()
        shard_rows = 0
        shard_random_reservoir = {"seen": 0, "samples": []}
        print(
            f"[{file_number}/{len(files)}] Scanning {file_path.name} "
            f"({file_size / (1024 * 1024):.1f} MiB; metadata rows={metadata_rows:,})",
            flush=True,
        )

        batch_number = 0
        for batch in parquet_file.iter_batches(
            batch_size=args.batch_size,
            columns=list(REQUIRED_COLUMNS),
            use_threads=True,
        ):
            batch_number += 1
            rows = batch.to_pylist()
            if args.max_rows_per_shard:
                remaining = args.max_rows_per_shard - shard_rows
                if remaining <= 0:
                    break
                rows = rows[:remaining]

            duplicate_rows += update_fingerprints(connection, rows)

            for record in rows:
                shard_rows += 1
                total_rows += 1
                flags = get_record_flags(record)
                for flag in flags:
                    global_flags[flag] += 1
                    shard_flags[flag] += 1
                    flag_sample = sample_preview(record, file_path.name, shard_rows)
                    add_reservoir_sample(
                        flag_reservoirs.setdefault(flag, {"seen": 0, "samples": []}),
                        {"sample_reason": flag, **flag_sample},
                        args.flag_samples_per_category,
                        rng,
                    )

                for field in REQUIRED_COLUMNS:
                    observe_field(global_fields[field], record.get(field))
                    observe_field(shard_fields[field], record.get(field))

                preview = sample_preview(record, file_path.name, shard_rows)
                add_reservoir_sample(
                    shard_random_reservoir,
                    preview,
                    args.samples_per_shard,
                    rng,
                )

                q_text = safe_text(record.get("question")).strip()
                r_text = safe_text(record.get("response")).strip()
                language_text = (q_text + "\n" + r_text).strip()
                if language_text:
                    add_reservoir_sample(
                        language_reservoir,
                        {"text": language_text[:2_000]},
                        args.language_sample_size,
                        language_rng,
                    )

            if batch_number % 100 == 0:
                print(
                    f"    scanned {shard_rows:,} rows from this shard; "
                    f"{total_rows:,} total rows so far",
                    flush=True,
                )

            if args.max_rows_per_shard and shard_rows >= args.max_rows_per_shard:
                break

        random_samples_by_shard.extend(shard_random_reservoir["samples"])
        shard_summaries.append(
            {
                "file": file_path.name,
                "file_size_bytes": file_size,
                "metadata_row_count": metadata_rows,
                "audited_row_count": shard_rows,
                "complete_shard_scanned": (
                    not args.max_rows_per_shard or shard_rows >= metadata_rows
                ),
                "field_stats": {
                    key: finalize_field_stats(value)
                    for key, value in shard_fields.items()
                },
                "flag_counts": dict(shard_flags.most_common()),
            }
        )
        print(
            f"    finished {file_path.name}: audited {shard_rows:,} rows; "
            f"{total_rows:,} total rows so far",
            flush=True,
        )

    language_samples = language_reservoir["samples"]
    language_summary = classify_language_sample(language_samples)

    return {
        "total_rows": total_rows,
        "duplicate_rows_seen_during_scan": duplicate_rows,
        "fields": {
            key: finalize_field_stats(value)
            for key, value in global_fields.items()
        },
        "flag_counts": dict(global_flags.most_common()),
        "shards": shard_summaries,
        "schema_by_shard": schema_by_shard,
        "schema_mismatches": missing_required_schema,
        "uniform_review_samples": random_samples_by_shard,
        "flagged_review_samples": [
            {"flag": flag, **sample}
            for flag, state in sorted(flag_reservoirs.items())
            for sample in state["samples"]
        ],
        "language_profile": language_summary,
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(
            f"ERROR: input directory does not exist: {input_dir}\n"
            "Download the dataset first or pass the correct --input-dir.",
            file=sys.stderr,
        )
        return 2

    files = sorted(input_dir.glob("train-*.parquet"))
    if not files:
        files = sorted(input_dir.glob("*.parquet"))
    if not files:
        print(f"ERROR: no Parquet files found in {input_dir}", file=sys.stderr)
        return 2

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parquet shards found: {len(files)}")
    print(f"Batch size: {args.batch_size:,}")
    if args.max_rows_per_shard:
        print(
            "WARNING: this is a partial smoke-test run; "
            f"capped at {args.max_rows_per_shard:,} rows per shard."
        )
    if langid is None:
        print("Language profile: unavailable (optional package 'langid' not installed).")

    if args.keep_index:
        index_path = output_dir / "fingerprints.sqlite3"
        if index_path.exists():
            index_path.unlink()
        temp_directory = None
    else:
        temp_directory = tempfile.TemporaryDirectory(
            prefix=".audit-index-", dir=output_dir
        )
        index_path = Path(temp_directory.name) / "fingerprints.sqlite3"

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(index_path)
        initialize_index(connection)
        scan_result = scan_files(args, files, connection)
        fingerprints = summarize_fingerprints(connection, scan_result["total_rows"])

        schema_signatures = {
            tuple(names) for names in scan_result["schema_by_shard"].values()
        }
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "Prarabdha/indian-legal-supervised-fine-tuning-data",
            "dataset_revision": args.dataset_revision,
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "python_version": platform.python_version(),
            "pyarrow_version": pyarrow.__version__,
            "parquet_files_found": len(files),
            "batch_size": args.batch_size,
            "samples_per_shard_requested": args.samples_per_shard,
            "flag_samples_per_category_requested": args.flag_samples_per_category,
            "language_sample_size_requested": args.language_sample_size,
            "max_rows_per_shard": args.max_rows_per_shard,
            "partial_run": bool(args.max_rows_per_shard),
            "seed": args.seed,
            "total_rows_audited": scan_result["total_rows"],
            "expected_metadata_rows_sum": sum(
                shard["metadata_row_count"] for shard in scan_result["shards"]
            ),
            "all_shards_fully_scanned": all(
                shard["complete_shard_scanned"] for shard in scan_result["shards"]
            ),
            "schema_identical_across_shards": len(schema_signatures) == 1,
            "fields": scan_result["fields"],
            "flag_counts": scan_result["flag_counts"],
            "duplicate_analysis": fingerprints,
            "language_profile": scan_result["language_profile"],
            "shards": scan_result["shards"],
            "schema_by_shard": scan_result["schema_by_shard"],
            "notes": [
                "This audit is read-only; source Parquet files are never changed.",
                "Heuristic flags are review signals, not automatic pass/fail decisions.",
                "Form-feed characters (\\f) are counted separately as possible PDF page breaks and are not included in control-character flags.",
                "Placeholder-response, statement-like-question, and missing-legal-reference-number flags are heuristics that require review.",
                "No LLM was used to label or certify legal correctness in this stage.",
                "Language results are sample estimates when the optional langid package is available.",
                "For a final full-corpus report, rerun without --max-rows-per-shard.",
            ],
        }

        (output_dir / "audit_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_jsonl(output_dir / "review_samples.jsonl", scan_result["uniform_review_samples"])
        write_jsonl(output_dir / "flagged_samples.jsonl", scan_result["flagged_review_samples"])

        print("\nAudit finished.")
        print(f"Rows audited: {scan_result['total_rows']:,}")
        print(f"Full-record duplicate occurrences: {fingerprints['normalized_duplicate_full_record_occurrences']:,}")
        print(f"Questions with multiple distinct responses: {fingerprints['questions_with_multiple_distinct_responses']:,}")
        print(f"Summary: {output_dir / 'audit_summary.json'}")
        print(f"Random samples: {output_dir / 'review_samples.jsonl'}")
        print(f"Flagged samples: {output_dir / 'flagged_samples.jsonl'}")
        if not args.keep_index:
            print("Temporary fingerprint index will be removed automatically.")
        return 0
    except Exception as exc:
        print(f"ERROR while auditing: {exc}", file=sys.stderr)
        raise
    finally:
        if connection is not None:
            connection.close()
        if temp_directory is not None:
            temp_directory.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
