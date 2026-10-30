"""Build a reproducible, human-reviewable language validation sample.

This utility only samples and labels records for evaluation. It never assigns
PASS/QUARANTINE/REJECT outcomes, filters the source, or writes to raw data.
Run from the repository root, for example:

  uv run python scripts/data_pipeline/audit/validate_language.py \
    --sample-size 10000 --seed 42 \
    --output-dir data/reports/language-validation-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_pipeline.audit.run_audit import (  # noqa: E402
    CONTROL_CHARACTERS,
    REQUIRED_COLUMNS,
    get_record_flags,
)

try:
    import langid
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise RuntimeError(
        "langid is required for language validation; install project dependencies first"
    ) from exc


DEFAULT_INPUT_DIR = Path(
    "data/raw/indian-legal-supervised-fine-tuning-data/data"
)
DEFAULT_REVISION = "fe58e87d8393ddd0b8e4224e3eec086c37f1f78e"
DEFAULT_SAMPLE_SIZE = 10_000
CORE_SAMPLE_FRACTION = 0.8
CLASSIFIER_TEXT_LIMIT = 2_000
SHORT_TEXT_CHARS = 40
LONG_TEXT_CHARS = 1_000

LEGAL_CITATION = re.compile(
    r"\b(?:article|section|rule|regulation|clause|sub-section)\s*"
    r"(?:\d|[ivxlcdm]+|[([]|of\b)",
    re.IGNORECASE,
)
NAME_LIKE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b")
QUOTE_MARKS = ('"', "'", "“", "”", "‘", "’", "«", "»")

REVIEW_FIELDS = {
    "human_question_language": {
        "description": "Human label for the question field using a language code; use mixed or uncertain when appropriate.",
        "allowed_values": "ISO 639-1/639-3 language code, mixed, uncertain",
    },
    "human_response_language": {
        "description": "Human label for the response field using a language code; use mixed or uncertain when appropriate.",
        "allowed_values": "ISO 639-1/639-3 language code, mixed, uncertain",
    },
    "human_mixed_language_status": {
        "description": "Whether either field contains multiple languages.",
        "allowed_values": ["none", "question", "response", "both", "uncertain"],
    },
    "question_genuinely_english": {
        "description": "Whether the question is genuinely English, including meaningful legal-language content rather than names or short quotations alone.",
        "allowed_values": ["yes", "no", "partial", "uncertain"],
    },
    "response_genuinely_english": {
        "description": "Whether the response is genuinely English, including meaningful legal-language content rather than names or short quotations alone.",
        "allowed_values": ["yes", "no", "partial", "uncertain"],
    },
    "reviewer_id": {"description": "Reviewer identifier or initials.", "allowed_values": "free text"},
    "review_notes": {"description": "Brief rationale for uncertain or mixed cases.", "allowed_values": "free text"},
}


class LanguageClassifier(Protocol):
    def classify(self, text: str) -> tuple[str, float]: ...


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def language_classifier_version() -> str:
    try:
        return version("langid")
    except PackageNotFoundError:  # pragma: no cover - import already succeeded
        return "unknown"


def list_parquet_shards(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No Parquet shards found in {input_dir}")
    return files


def allocate_proportionally(total: int, weights: list[int]) -> list[int]:
    """Allocate an integer sample across shards by metadata row counts."""
    if total <= 0 or not weights:
        return [0 for _ in weights]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [0 for _ in weights]
    exact = [total * weight / weight_sum for weight in weights]
    result = [math.floor(value) for value in exact]
    remainder = total - sum(result)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - result[index]), index),
    )
    for index in order[:remainder]:
        result[index] += 1
    return result


def _field_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _script_names(text: str) -> set[str]:
    if text.isascii():
        return {"LATIN"} if any(char.isalpha() for char in text) else set()
    scripts: set[str] = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name:
            scripts.add(name.split()[0])
    return scripts


def heuristic_strata(record: dict[str, Any]) -> list[str]:
    """Candidate strata for targeted review; these are not language labels."""
    question = _field_text(record.get("question"))
    response = _field_text(record.get("response"))
    combined = f"{question}\n{response}"
    nonblank = [text for text in (question, response) if text.strip()]
    strata: list[str] = []

    if any(len(text.strip()) < SHORT_TEXT_CHARS for text in nonblank):
        strata.append("short_text_under_40_chars")
    if any(len(text) >= LONG_TEXT_CHARS for text in (question, response)):
        strata.append("long_text_at_least_1000_chars")

    q_scripts = _script_names(question)
    r_scripts = _script_names(response)
    all_scripts = q_scripts | r_scripts
    if any(script != "LATIN" for script in all_scripts):
        strata.append("non_latin_script_candidate")
    if len(q_scripts) > 1 or len(r_scripts) > 1 or (
        q_scripts and r_scripts and q_scripts != r_scripts
    ):
        strata.append("mixed_language_looking")

    if NAME_LIKE.search(combined):
        strata.append("name_like_capitalized_sequence")
    if LEGAL_CITATION.search(combined):
        strata.append("legal_citation")
    if any(mark in combined for mark in QUOTE_MARKS):
        strata.append("quotation_mark")
    if CONTROL_CHARACTERS.search(question) or CONTROL_CHARACTERS.search(response):
        strata.append("suspicious_control_character")
    if "\ufffd" in question or "\ufffd" in response:
        strata.append("replacement_character")
    source_fields = [_field_text(record.get(field)) for field in REQUIRED_COLUMNS]
    if any(CONTROL_CHARACTERS.search(text) or "\ufffd" in text for text in source_fields):
        strata.append("audit_control_or_replacement_flag")
    if any("\f" in text for text in source_fields):
        strata.append("form_feed")
    return sorted(set(strata))


def _make_sample_record(
    record: dict[str, Any],
    shard: str,
    row_number: int,
    dataset_revision: str,
) -> dict[str, Any]:
    context = record.get("context")
    return {
        "source_row_id": f"{dataset_revision}:{shard}:{row_number}",
        "source_shard": shard,
        "source_row_number_1_based": row_number,
        "dataset_revision": dataset_revision,
        "question": record.get("question"),
        "response": record.get("response"),
        # Kept in the local review file only for source provenance. The
        # classifier receives question and response in separate calls.
        "context": context,
        "context_role": "provenance_only_do_not_use_for_language_label",
        "field_lengths_chars": {
            field: len(_field_text(record.get(field))) for field in REQUIRED_COLUMNS
        },
        "candidate_strata": heuristic_strata(record),
    }


def _add_reservoir(
    state: dict[str, Any],
    item: dict[str, Any],
    capacity: int,
    rng: random.Random,
) -> None:
    if capacity <= 0:
        return
    state["seen"] += 1
    seen = state["seen"]
    values = state["items"]
    if len(values) < capacity:
        values.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < capacity:
        values[replacement] = item


def classify_field(
    value: Any,
    classifier: LanguageClassifier = langid,
    text_limit: int = CLASSIFIER_TEXT_LIMIT,
) -> dict[str, Any]:
    """Classify one field and preserve langid's raw score, not a probability."""
    if value is None:
        return {
            "status": "missing",
            "language": None,
            "classifier_score": None,
            "score_is_probability": False,
            "input_characters": 0,
            "classified_characters": 0,
            "input_truncated": False,
        }
    if not isinstance(value, str):
        return {
            "status": "non_string",
            "language": None,
            "classifier_score": None,
            "score_is_probability": False,
            "input_characters": None,
            "classified_characters": 0,
            "input_truncated": False,
        }
    text = value.strip()
    if not text:
        return {
            "status": "blank",
            "language": None,
            "classifier_score": None,
            "score_is_probability": False,
            "input_characters": len(value),
            "classified_characters": 0,
            "input_truncated": False,
        }

    classifier_text = text[:text_limit]
    try:
        language, score = classifier.classify(classifier_text)
    except Exception as exc:  # Report a row-level failure without aborting sampling.
        return {
            "status": "classification_error",
            "language": None,
            "classifier_score": None,
            "score_is_probability": False,
            "input_characters": len(text),
            "classified_characters": len(classifier_text),
            "input_truncated": len(text) > len(classifier_text),
            "error_type": type(exc).__name__,
        }
    return {
        "status": "classified",
        "language": str(language),
        "classifier_score": float(score),
        "score_is_probability": False,
        "input_characters": len(text),
        "classified_characters": len(classifier_text),
        "input_truncated": len(text) > len(classifier_text),
    }


def classify_record_fields(
    record: dict[str, Any], classifier: LanguageClassifier = langid
) -> dict[str, dict[str, Any]]:
    """Classify question and response independently; never concatenate them."""
    return {
        "question": classify_field(record.get("question"), classifier),
        "response": classify_field(record.get("response"), classifier),
    }


def _candidate_capacities(targeted_slots: int) -> tuple[int, int]:
    if targeted_slots <= 0:
        return 0, 0
    per_stratum = max(25, min(500, targeted_slots * 2))
    language_candidates = max(500, min(5_000, targeted_slots * 2))
    return per_stratum, language_candidates


def _stratum_rngs(seed: int, names: list[str]) -> dict[str, random.Random]:
    return {name: random.Random(_stable_seed(seed, name)) for name in names}


def collect_validation_sample(
    input_dir: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
    batch_size: int = 2_048,
    dataset_revision: str = DEFAULT_REVISION,
    classifier: LanguageClassifier = langid,
    expected_shard_count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read all shards in batches and make core plus supplemental samples."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if expected_shard_count is not None and expected_shard_count < 1:
        raise ValueError("expected_shard_count must be at least 1")

    files = list_parquet_shards(input_dir)
    if expected_shard_count is not None and len(files) != expected_shard_count:
        raise ValueError(
            f"Expected {expected_shard_count} Parquet shards in {input_dir}; "
            f"found {len(files)}"
        )
    metadata_rows = [pq.ParquetFile(path).metadata.num_rows for path in files]
    metadata_total = sum(metadata_rows)
    if metadata_total < 1:
        raise ValueError("The Parquet shards contain no rows")

    effective_target = min(sample_size, metadata_total)
    core_target = min(
        effective_target,
        max(1, math.floor(effective_target * CORE_SAMPLE_FRACTION)),
    )
    targeted_slots = effective_target - core_target
    core_allocations = allocate_proportionally(core_target, metadata_rows)
    stratum_capacity, language_candidate_capacity = _candidate_capacities(
        targeted_slots
    )

    candidate_names = [
        "short_text_under_40_chars",
        "long_text_at_least_1000_chars",
        "mixed_language_looking",
        "name_like_capitalized_sequence",
        "legal_citation",
        "quotation_mark",
        "suspicious_control_character",
        "replacement_character",
        "audit_control_or_replacement_flag",
        "form_feed",
        "non_latin_script_candidate",
    ]
    core_states: dict[str, dict[str, Any]] = {}
    for path, capacity in zip(files, core_allocations, strict=True):
        core_states[path.name] = {
            "seen": 0,
            "items": [],
            "capacity": capacity,
            "rng": random.Random(_stable_seed(seed, f"core:{path.name}")),
        }

    category_states = {
        name: {"seen": 0, "items": []} for name in candidate_names
    }
    category_rngs = _stratum_rngs(seed, candidate_names)
    rows_scanned_by_shard: dict[str, int] = {}
    schema_by_shard: dict[str, list[str]] = {}
    total_rows_scanned = 0

    for file_number, (path, expected_rows) in enumerate(
        zip(files, metadata_rows, strict=True), start=1
    ):
        parquet_file = pq.ParquetFile(path)
        schema = list(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_COLUMNS) - set(schema))
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )
        schema_by_shard[path.name] = schema
        shard_rows = 0
        print(
            f"[{file_number}/{len(files)}] Sampling {path.name} "
            f"(metadata rows={expected_rows:,})",
            flush=True,
        )
        core_state = core_states[path.name]
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(REQUIRED_COLUMNS),
            use_threads=True,
        ):
            for record in batch.to_pylist():
                shard_rows += 1
                total_rows_scanned += 1
                item = _make_sample_record(
                    record, path.name, shard_rows, dataset_revision
                )
                _add_reservoir(
                    core_state,
                    item,
                    core_state["capacity"],
                    core_state["rng"],
                )
                for stratum in item["candidate_strata"]:
                    state = category_states[stratum]
                    _add_reservoir(
                        state,
                        item,
                        language_candidate_capacity
                        if stratum == "non_latin_script_candidate"
                        else stratum_capacity,
                        category_rngs[stratum],
                    )
        if shard_rows != expected_rows:
            raise RuntimeError(
                f"Read {shard_rows} rows from {path.name}; metadata reports {expected_rows}"
            )
        rows_scanned_by_shard[path.name] = shard_rows

    core_sample: list[dict[str, Any]] = []
    for path, allocation, population in zip(files, core_allocations, metadata_rows, strict=True):
        state = core_states[path.name]
        for item in state["items"]:
            copied = dict(item)
            copied["sampling_basis"] = ["uniform_core"]
            copied["uniform_core_inclusion_probability"] = (
                allocation / population if population else None
            )
            copied["supplemental_selection_strata"] = []
            core_sample.append(copied)

    # Classify the core probability sample and bounded candidate reservoirs.
    # Candidate predictions are used only to select human-review cases; they
    # are not included as corpus estimates.
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for state in category_states.values():
        for item in state["items"]:
            existing = candidate_by_id.get(item["source_row_id"])
            if existing is None:
                candidate_by_id[item["source_row_id"]] = dict(item)
            else:
                existing["candidate_strata"] = sorted(
                    set(existing["candidate_strata"]) | set(item["candidate_strata"])
                )

    predictions_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for item in core_sample:
        predictions_by_id[item["source_row_id"]] = classify_record_fields(
            item, classifier
        )
    for source_row_id, item in candidate_by_id.items():
        if source_row_id not in predictions_by_id:
            predictions_by_id[source_row_id] = classify_record_fields(item, classifier)
        item["predictions"] = predictions_by_id[source_row_id]
        q_language = item["predictions"]["question"].get("language")
        r_language = item["predictions"]["response"].get("language")
        if q_language:
            item["candidate_strata"].append(f"predicted_question_language:{q_language}")
        if r_language:
            item["candidate_strata"].append(f"predicted_response_language:{r_language}")
        if q_language and r_language and q_language != r_language:
            item["candidate_strata"].append("question_response_prediction_disagreement")
        item["candidate_strata"] = sorted(set(item["candidate_strata"]))

    selected_ids = {item["source_row_id"] for item in core_sample}
    supplemental: list[dict[str, Any]] = []
    category_order: list[str] = []

    # Prioritize field-specific predicted labels (including non-English labels)
    # and high-value heuristic strata. The final report labels this as targeted.
    predicted_categories = sorted(
        {
            stratum
            for item in candidate_by_id.values()
            for stratum in item["candidate_strata"]
            if stratum.startswith("predicted_question_language:")
            or stratum.startswith("predicted_response_language:")
        },
        key=lambda value: (value.endswith(":en"), value),
    )
    heuristic_priority = [
        "mixed_language_looking",
        "non_latin_script_candidate",
        "suspicious_control_character",
        "replacement_character",
        "audit_control_or_replacement_flag",
        "short_text_under_40_chars",
        "long_text_at_least_1000_chars",
        "legal_citation",
        "name_like_capitalized_sequence",
        "quotation_mark",
        "form_feed",
        "question_response_prediction_disagreement",
    ]
    category_order.extend(predicted_categories)
    category_order.extend(
        name
        for name in heuristic_priority
        if any(name in item["candidate_strata"] for item in candidate_by_id.values())
    )
    category_order = list(dict.fromkeys(category_order))
    category_quota = max(
        1,
        min(50, math.ceil(targeted_slots / max(1, len(category_order)))),
    )

    candidate_rng = random.Random(_stable_seed(seed, "supplemental-selection"))
    candidates_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in category_order:
        candidates = [
            item
            for item in candidate_by_id.values()
            if category in item["candidate_strata"]
            and item["source_row_id"] not in selected_ids
        ]
        candidate_rng.shuffle(candidates)
        candidates_by_category[category] = candidates

    selected_per_category: Counter[str] = Counter()
    for category in category_order:
        for item in candidates_by_category[category]:
            if len(supplemental) >= targeted_slots:
                break
            source_row_id = item["source_row_id"]
            if source_row_id in selected_ids:
                continue
            copied = dict(item)
            copied["sampling_basis"] = ["supplemental_targeted"]
            copied["uniform_core_inclusion_probability"] = None
            copied["supplemental_selection_strata"] = [category]
            supplemental.append(copied)
            selected_ids.add(source_row_id)
            selected_per_category[category] += 1
            if selected_per_category[category] >= category_quota:
                break

    if len(supplemental) < targeted_slots:
        remaining_candidates = [
            item
            for item in candidate_by_id.values()
            if item["source_row_id"] not in selected_ids
        ]
        candidate_rng.shuffle(remaining_candidates)
        for item in remaining_candidates[: targeted_slots - len(supplemental)]:
            copied = dict(item)
            copied["sampling_basis"] = ["supplemental_targeted"]
            copied["uniform_core_inclusion_probability"] = None
            copied["supplemental_selection_strata"] = ["targeted_pool_fill"]
            supplemental.append(copied)
            selected_ids.add(item["source_row_id"])

    sample = core_sample + supplemental
    sample.sort(
        key=lambda item: (
            item["source_shard"],
            item["source_row_number_1_based"],
        )
    )

    # Re-use predictions selected during core/candidate handling. If a
    # supplemental record is also in the core, it is already represented there.
    for item in sample:
        context_text = _field_text(item.get("context"))
        item["context_sha256"] = hashlib.sha256(
            context_text.encode("utf-8")
        ).hexdigest()
        item["audit_flags"] = get_record_flags(item)
        item["question_prediction"], item["response_prediction"] = (
            predictions_by_id[item["source_row_id"]]["question"],
            predictions_by_id[item["source_row_id"]]["response"],
        )
        item["human_question_language"] = ""
        item["human_response_language"] = ""
        item["human_mixed_language_status"] = ""
        item["question_genuinely_english"] = ""
        item["response_genuinely_english"] = ""
        item["reviewer_id"] = ""
        item["review_notes"] = ""

    sample_counts_by_shard = Counter(item["source_shard"] for item in sample)
    prediction_counts: dict[str, dict[str, int]] = {"question": {}, "response": {}}
    prediction_status_counts: dict[str, dict[str, int]] = {
        "question": {},
        "response": {},
    }
    for item in sample:
        for field in ("question", "response"):
            prediction = item[f"{field}_prediction"]
            if prediction.get("language"):
                language_counts = prediction_counts[field]
                language_counts[prediction["language"]] = (
                    language_counts.get(prediction["language"], 0) + 1
                )
            status_counts = prediction_status_counts[field]
            status_counts[prediction["status"]] = (
                status_counts.get(prediction["status"], 0) + 1
            )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": dataset_revision,
        "input_directory": str(input_dir),
        "shard_count": len(files),
        "expected_shard_count": expected_shard_count,
        "shards": [
            {
                "file": path.name,
                "metadata_row_count": count,
                "rows_scanned": rows_scanned_by_shard[path.name],
                "sample_rows": sample_counts_by_shard[path.name],
                "schema": schema_by_shard[path.name],
                "core_sample_quota": allocation,
                "core_inclusion_probability": allocation / count if count else None,
            }
            for path, count, allocation in zip(
                files, metadata_rows, core_allocations, strict=True
            )
        ],
        "metadata_total_rows": metadata_total,
        "total_rows_scanned": total_rows_scanned,
        "requested_sample_pairs": sample_size,
        "sample_size_actual": len(sample),
        "core_probability_sample_count": len(core_sample),
        "supplemental_targeted_sample_count": len(supplemental),
        "sample_seed": seed,
        "sampling_design": {
            "core_method": "Fixed-seed independent reservoir sample within each shard, with shard quotas allocated proportionally to Parquet metadata row counts. The reservoir considers every row in each shard; it is not limited to the first rows.",
            "core_target_count": core_target,
            "core_inclusion_probability_by_shard": "core_sample_quota / shard_metadata_row_count; recorded per sample and per shard",
            "targeted_method": "Supplemental cases are selected from fixed-seed, bounded reservoir samples of heuristic strata and candidate langid prediction labels. Per-row stratum membership and selection reason are recorded. These targeted cases have no single overall inclusion probability and must not be treated as a simple random sample.",
            "candidate_reservoir_capacity_per_heuristic_stratum": stratum_capacity,
            "candidate_reservoir_capacity_non_latin_script": language_candidate_capacity,
            "candidate_strata_population_counts": {
                name: state["seen"] for name, state in category_states.items()
            },
            "candidate_strata_reservoir_counts": {
                name: len(state["items"]) for name, state in category_states.items()
            },
            "candidate_strata_selected_counts": dict(selected_per_category),
            "heuristic_strata_are_not_language_labels": True,
        },
        "classifier": {
            "name": "langid",
            "version": language_classifier_version(),
            "fields_classified_separately": ["question", "response"],
            "maximum_characters_per_field": CLASSIFIER_TEXT_LIMIT,
            "minimum_field_length": 0,
            "score_name": "classifier_score",
            "score_is_probability": False,
            "threshold": None,
            "note": "The raw langid score is not a calibrated probability or confidence percentage. No confidence threshold is defined.",
            "candidate_pool_screening_rows": len(predictions_by_id),
        },
        "sample_prediction_counts_only_not_corpus_estimates": prediction_counts,
        "sample_prediction_status_counts": prediction_status_counts,
        "outcomes_assigned": False,
        "raw_data_modified": False,
        "context_sent_to_external_service": False,
        "context_use": "Context is copied to the local review file for provenance only. It is not sent to an LLM or external API and is not used for language predictions.",
        "limitations": [
            "The 5,000-text language estimate in the prior audit is not used to select or classify rows.",
            "Combined sample prediction counts are not a full-corpus language census; supplemental targeted cases are deliberately overrepresented.",
            "The core probability sample is identifiable separately. Use its recorded shard-specific inclusion probabilities for population-oriented analysis; the supplemental sample is for error discovery and review coverage.",
            "Langid returns one predicted label and a raw classifier score. The score is not calibrated as a probability, and no confidence threshold is proposed.",
            "Short text, legal names, citations, quotations, code-switching, transliteration, and OCR damage can affect predictions.",
            "Script and text-pattern strata are candidate-finding heuristics and can include false positives.",
            "Human review is required before approving a language filter or any automatic language-based outcome.",
        ],
    }
    return summary, sample


def review_template_document() -> dict[str, Any]:
    return {
        "instructions": [
            "Label question and response independently using only the corresponding field text.",
            "The context field is present for provenance only. Do not use it to decide the language label or whether the question/response is English.",
            "Use ISO 639-1 or ISO 639-3 codes where possible; use mixed for a field with meaningful multiple-language content and uncertain when you cannot decide.",
            "For genuinely-English labels, yes means the field is substantively English; no means it is not; partial means English mixed with meaningful non-English content; uncertain means unresolved.",
            "A name, citation, Latin legal phrase, or short quotation alone does not make an otherwise non-English field genuinely English.",
        ],
        "fields": REVIEW_FIELDS,
        "review_is_not_an_automatic_filter": True,
    }


def render_sampling_report(summary: dict[str, Any]) -> str:
    strata_counts = summary["sampling_design"]["candidate_strata_population_counts"]
    selected_strata_counts = summary["sampling_design"][
        "candidate_strata_selected_counts"
    ]
    shard_lines = "\n".join(
        f"| `{shard['file']}` | {shard['metadata_row_count']:,} | "
        f"{shard['sample_rows']} | {shard['core_sample_quota']} |"
        for shard in summary["shards"]
    )
    stratum_lines = "\n".join(
        f"- `{name}`: {count:,} candidates encountered; "
        f"{summary['sampling_design']['candidate_strata_reservoir_counts'][name]:,} "
        "retained in its bounded reservoir."
        for name, count in sorted(strata_counts.items())
    )
    selected_stratum_lines = "\n".join(
        f"- `{name}`: {count:,} supplemental cases selected."
        for name, count in sorted(selected_strata_counts.items())
    ) or "- No targeted cases were selected."
    return f"""# Language validation sample report

## Scope

Dataset revision: `{summary['dataset_revision']}`<br>
Rows represented by shard metadata: {summary['metadata_total_rows']:,}<br>
Rows scanned in bounded batches: {summary['total_rows_scanned']:,}<br>
Parquet shards: {summary['shard_count']}<br>
Requested review pairs: {summary['requested_sample_pairs']:,}<br>
Actual review pairs: {summary['sample_size_actual']:,}<br>
Fixed seed: `{summary['sample_seed']}`

This is a language-classifier validation sample, not a cleaner. It assigns no PASS, QUARANTINE, or REJECT outcome, changes no source data, and creates no training export.

## Sampling procedure

The core portion contains {summary['core_probability_sample_count']:,} rows. It uses a fixed-seed reservoir sample within each shard; per-shard quotas are proportional to metadata row counts. Every row in each shard is considered, so selection is not limited to shard prefixes. The exact core inclusion probability for a shard is its core quota divided by its metadata row count, recorded in the summary and per-sample record.

The remaining {summary['supplemental_targeted_sample_count']:,} rows, when available, are supplemental targeted cases from bounded reservoirs for predicted language labels and text strata. These cover short/long text, mixed-script-looking text, name-like capitalized sequences, citations, quotations, control/replacement characters, and form-feeds when present. Their selection method and strata are recorded per row. Because strata overlap and the supplemental cases are deliberately selected, they have no single overall inclusion probability and must not be combined with the core sample as though the result were a simple random sample.

| Shard | Metadata rows | Final sample rows | Core quota |
|---|---:|---:|---:|
{shard_lines}

Full-scan candidate counts and retained reservoir counts:

{stratum_lines}

Targeted selection reasons:

{selected_stratum_lines}

## Classification

`langid` version `{summary['classifier']['version']}` classifies `question` and `response` separately, using up to {summary['classifier']['maximum_characters_per_field']:,} characters from each nonblank field. Missing and blank fields are recorded without classification. Short fields are still classified. No text from `context` is passed to the classifier. The raw classifier output is stored as a **score**, not a calibrated probability or confidence percentage; no confidence threshold is set.

The prediction counts in `language_classification_summary.json` describe only the final sample. Supplemental cases are overrepresented by design, so those combined counts are not corpus language estimates. The prior audit's approximately 5,000 combined-text sample is neither used here nor treated as a census.

## Human review steps

1. Open `language_review.jsonl` and review the `question` and `response` fields independently. The adjacent `context` value is present for source provenance only; do not use it to label the language or Englishness of either field.
2. Fill `human_question_language` and `human_response_language` with an ISO language code, `mixed`, or `uncertain`.
3. Fill `human_mixed_language_status` with `none`, `question`, `response`, `both`, or `uncertain`.
4. Fill `question_genuinely_english` and `response_genuinely_english` with `yes`, `no`, `partial`, or `uncertain`; add reviewer ID and notes for ambiguous cases.
5. Preserve `source_row_id`, predictions, scores, and sampling fields. Do not convert classifier scores to percentages or apply an outcome/filter based on these predictions.
6. Use `evaluate_language_review.py` on the completed JSONL to produce metrics and error examples. Review the probability-core results separately from the supplemental targeted cases.

## Limitations

- The classifier emits one label per field and may mishandle short legal phrases, names, Latin phrases, quotations, mixed-language/code-switched text, transliteration, and OCR-corrupted strings.
- The heuristic strata identify candidates, not confirmed mixed language, names, legal defects, or corrupted text.
- Human labels require a documented adjudication process. These metrics validate a classifier against the reviewed sample; they do not certify legal correctness or authorize automatic language-based exclusion by themselves.
- Do not interpret this sample as a full-corpus language census. Any V1 language decision still requires explicit approval of the classifier, per-field thresholds, mixed-language handling, and stratified human-validation procedure.
"""


def write_validation_outputs(
    output_dir: Path, summary: dict[str, Any], sample: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {output_dir}"
        )
    summary_path = output_dir / "language_classification_summary.json"
    review_path = output_dir / "language_review.jsonl"
    template_path = output_dir / "language_review_template.json"
    report_path = output_dir / "language_sampling_report.md"

    summary["output_files"] = {
        "classification_summary": summary_path.name,
        "human_review_file": review_path.name,
        "review_template": template_path.name,
        "sampling_report": report_path.name,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with review_path.open("w", encoding="utf-8") as handle:
        for item in sample:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    template_path.write_text(
        json.dumps(review_template_document(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_sampling_report(summary), encoding="utf-8")


def ensure_output_directory_available(output_dir: Path) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {output_dir}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a human-review sample for per-field langid validation."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=33,
        help="Fail if this many Parquet shards are not found (default: 33).",
    )
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_directory_available(args.output_dir)
    summary, sample = collect_validation_sample(
        input_dir=args.input_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        dataset_revision=args.dataset_revision,
        expected_shard_count=args.expected_shards,
    )
    write_validation_outputs(args.output_dir, summary, sample)
    print(
        f"Wrote {len(sample):,} review pairs from {summary['shard_count']} shards "
        f"to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke test
    main()
