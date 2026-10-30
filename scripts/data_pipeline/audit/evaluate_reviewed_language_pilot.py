"""Evaluate a user-reviewed pilot and inspect empirical langid score floors.

Question and response fields are classified separately with the same
``classify_field`` implementation and 2,000-character limit used by Stage 3.
The original predictions stay intact; current-run predictions use distinct
fields in the evaluation JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import langid
import langid.langid as langid_impl

from scripts.data_pipeline.audit.evaluate_language_review import (
    evaluate_records,
    normalize_label,
    read_jsonl,
)
from scripts.data_pipeline.audit.validate_language import (
    CLASSIFIER_TEXT_LIMIT,
    classify_field,
    language_classifier_version,
)

DEFAULT_DIR = Path("data/reports/language-validation-pilot-v1")
DEFAULT_REVIEWED_CSV = DEFAULT_DIR / "language_review_pilot_user_reviewed_v1.csv"
DEFAULT_REVIEWED_JSONL = DEFAULT_DIR / "language_review_pilot_user_reviewed_v1.jsonl"
LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")
SUPPORTED_LANGUAGES = frozenset({"en", "hi"})
DEFAULT_SCORE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
MAX_ERROR_PREVIEW_CHARS = 220


def _selection_group(record: Mapping[str, Any]) -> str:
    basis = record.get("sampling_basis") or []
    if "uniform_core" in basis:
        return "uniform_core_probability_sample"
    if "supplemental_targeted" in basis:
        return "targeted_supplemental"
    return "unknown_sampling_group"


def _usable_score(prediction: Any) -> float | None:
    if not isinstance(prediction, Mapping):
        return None
    value = prediction.get("classifier_score")
    if (
        prediction.get("status") != "classified"
        or prediction.get("score_is_probability") is not False
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _truth_class(label: Any) -> str:
    normalized = normalize_label(label)
    if normalized in SUPPORTED_LANGUAGES:
        return "supported"
    if normalized in {"mixed", "uncertain", None}:
        return "ambiguous"
    if LANGUAGE_CODE.fullmatch(normalized):
        return "unsupported"
    return "ambiguous"


def _prediction_class(label: Any) -> str:
    normalized = normalize_label(label)
    if normalized in SUPPORTED_LANGUAGES:
        return "supported"
    if normalized in {"mixed", "uncertain", None}:
        return "unresolved"
    if LANGUAGE_CODE.fullmatch(normalized):
        return "unsupported"
    return "unresolved"


def _design_weight(record: Mapping[str, Any]) -> float:
    """Return inverse overall pilot inclusion probability for core records."""
    source_probability = record.get("uniform_core_inclusion_probability")
    pilot_selection = record.get("pilot_selection") or {}
    conditional_probability = pilot_selection.get(
        "pilot_conditional_inclusion_probability"
    )
    try:
        probability = float(source_probability) * float(conditional_probability)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing core inclusion probability for {record.get('source_row_id')}"
        ) from exc
    if not math.isfinite(probability) or probability <= 0 or probability > 1:
        raise ValueError(
            f"Invalid overall inclusion probability {probability!r} for {record.get('source_row_id')}"
        )
    return 1.0 / probability


def validate_reviewed_labels(
    rows_by_id: Mapping[str, Mapping[str, str]],
    records: Sequence[Mapping[str, Any]],
    template_path: Path,
) -> dict[str, Any]:
    """Validate the reviewed CSV without inferring or repairing any labels."""
    template = json.loads(template_path.read_text(encoding="utf-8"))
    fields = template["fields"]
    record_ids = {str(record.get("source_row_id")) for record in records}
    csv_ids = set(rows_by_id)
    if record_ids != csv_ids:
        raise ValueError("Reviewed CSV and pilot JSONL source_row_id sets do not match")

    label_fields = (
        "human_question_language",
        "human_response_language",
        "human_mixed_language_status",
        "question_genuinely_english",
        "response_genuinely_english",
        "reviewer_id",
        "review_notes",
    )
    missing_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    inconsistencies: list[dict[str, str]] = []
    for source_id, row in rows_by_id.items():
        for field in label_fields:
            if not row.get(field, "").strip():
                missing_counts[field] += 1
        for field in ("human_question_language", "human_response_language"):
            value = normalize_label(row.get(field))
            if value not in {"mixed", "uncertain"} and (
                value is None or not LANGUAGE_CODE.fullmatch(value)
            ):
                invalid_counts[field] += 1
        for field in (
            "human_mixed_language_status",
            "question_genuinely_english",
            "response_genuinely_english",
        ):
            allowed = fields[field]["allowed_values"]
            value = row.get(field, "").strip().lower()
            if value not in allowed:
                invalid_counts[field] += 1

        question = row.get("human_question_language", "").strip().lower()
        response = row.get("human_response_language", "").strip().lower()
        status = row.get("human_mixed_language_status", "").strip().lower()
        q_eng = row.get("question_genuinely_english", "").strip().lower()
        r_eng = row.get("response_genuinely_english", "").strip().lower()
        if status not in {"uncertain", ""}:
            q_status_mixed = status in {"question", "both"}
            r_status_mixed = status in {"response", "both"}
            if (question == "mixed") != q_status_mixed:
                inconsistencies.append(
                    {"source_row_id": source_id, "field": "question_mixed_status"}
                )
            if (response == "mixed") != r_status_mixed:
                inconsistencies.append(
                    {"source_row_id": source_id, "field": "response_mixed_status"}
                )
        if question == "mixed" and q_eng not in {"partial", "uncertain"}:
            inconsistencies.append(
                {"source_row_id": source_id, "field": "question_mixed_englishness"}
            )
        if response == "mixed" and r_eng not in {"partial", "uncertain"}:
            inconsistencies.append(
                {"source_row_id": source_id, "field": "response_mixed_englishness"}
            )

    return {
        "row_count": len(rows_by_id),
        "unique_source_row_ids": len(csv_ids),
        "missing_field_counts": dict(missing_counts),
        "invalid_field_value_counts": dict(invalid_counts),
        "semantic_consistency_issue_count": len(inconsistencies),
        "semantic_consistency_issues": inconsistencies,
        "language_label_counts": {
            field: dict(Counter(row.get(field, "").strip().lower() for row in rows_by_id.values()))
            for field in ("human_question_language", "human_response_language")
        },
        "mixed_status_counts": dict(
            Counter(row.get("human_mixed_language_status", "").strip().lower() for row in rows_by_id.values())
        ),
        "englishness_counts": {
            field: dict(Counter(row.get(field, "").strip().lower() for row in rows_by_id.values()))
            for field in ("question_genuinely_english", "response_genuinely_english")
        },
        "review_provenance_counts": dict(
            Counter(row.get("reviewer_id", "").strip() for row in rows_by_id.values())
        ),
    }


def candidate_score_floors(
    records: Sequence[Mapping[str, Any]],
    field: str,
    quantiles: Sequence[float] = DEFAULT_SCORE_QUANTILES,
) -> list[dict[str, Any]]:
    """Choose observed-score cutpoints from core scores, without using labels.

    Quantiles are diagnostic operating points, not recommended thresholds.
    The list is derived only from the probability-core score distribution.
    """
    scores = sorted(
        score
        for record in records
        if _selection_group(record) == "uniform_core_probability_sample"
        for score in [_usable_score(record.get(f"current_{field}_prediction"))]
        if score is not None
    )
    if not scores:
        return []
    candidates: list[dict[str, Any]] = [
        {"candidate": "no_floor", "quantile": None, "minimum_score": None}
    ]
    seen: set[float] = set()
    for quantile in quantiles:
        if not 0 <= quantile <= 1:
            raise ValueError("score quantiles must be between 0 and 1")
        index = round(quantile * (len(scores) - 1))
        floor = scores[index]
        if floor in seen:
            continue
        seen.add(floor)
        candidates.append(
            {
                "candidate": f"observed_core_q{quantile:g}",
                "quantile": quantile,
                "minimum_score": floor,
            }
        )
    return candidates


def analyze_score_floor(
    records: Sequence[Mapping[str, Any]],
    field: str,
    minimum_score: float | None,
    *,
    design_weighted: bool = False,
) -> dict[str, Any]:
    """Measure pass/reject/quarantine consequences against reviewed labels."""
    human_field = f"human_{field}_language"
    prediction_field = f"current_{field}_prediction"
    counts: Counter[str] = Counter()
    weighted: Counter[str] = Counter()
    truth_counts: Counter[str] = Counter()
    weighted_truth: Counter[str] = Counter()

    for record in records:
        truth = _truth_class(record.get(human_field))
        counts[f"truth_{truth}"] += 1
        weight = (
            _design_weight(record)
            if design_weighted
            and _selection_group(record) == "uniform_core_probability_sample"
            else 1.0
        )
        weighted[f"truth_{truth}"] += weight
        if truth != "ambiguous":
            truth_counts[truth] += 1
            weighted_truth[truth] += weight

        prediction = record.get(prediction_field) or {}
        score = _usable_score(prediction)
        if score is None:
            counts["quarantined_missing_or_invalid_score"] += 1
            weighted["quarantined_missing_or_invalid_score"] += weight
            if truth != "ambiguous":
                counts[f"{truth}_truth_quarantined_missing_or_invalid_score"] += 1
                weighted[f"{truth}_truth_quarantined_missing_or_invalid_score"] += weight
            continue
        if minimum_score is not None and score < minimum_score:
            counts["quarantined_below_score_floor"] += 1
            weighted["quarantined_below_score_floor"] += weight
            if truth != "ambiguous":
                counts[f"{truth}_truth_quarantined_below_score_floor"] += 1
                weighted[f"{truth}_truth_quarantined_below_score_floor"] += weight
            continue

        predicted_class = _prediction_class(prediction.get("language"))
        if predicted_class == "unresolved":
            counts["quarantined_unresolved_prediction"] += 1
            weighted["quarantined_unresolved_prediction"] += weight
            if truth != "ambiguous":
                counts[f"{truth}_truth_quarantined_unresolved_prediction"] += 1
                weighted[f"{truth}_truth_quarantined_unresolved_prediction"] += weight
        elif truth == "ambiguous":
            # Keep mixed/uncertain human labels out of binary error rates while
            # still accounting for all pilot rows that would be quarantined.
            continue
        elif predicted_class == "supported":
            counts["accepted_as_supported"] += 1
            weighted["accepted_as_supported"] += weight
            if truth == "unsupported":
                counts["unsupported_truth_predicted_supported_false_accept"] += 1
                weighted["unsupported_truth_predicted_supported_false_accept"] += weight
            else:
                counts["supported_truth_retained_as_supported"] += 1
                weighted["supported_truth_retained_as_supported"] += weight
        else:
            counts["accepted_as_unsupported"] += 1
            weighted["accepted_as_unsupported"] += weight
            if truth == "supported":
                counts["supported_truth_predicted_unsupported_false_reject"] += 1
                weighted["supported_truth_predicted_unsupported_false_reject"] += weight
            else:
                counts["unsupported_truth_predicted_unsupported_correct_reject"] += 1
                weighted["unsupported_truth_predicted_unsupported_correct_reject"] += weight

    def rate(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    unsupported_denom = weighted_truth["unsupported"] if design_weighted else truth_counts["unsupported"]
    supported_denom = weighted_truth["supported"] if design_weighted else truth_counts["supported"]
    table = weighted if design_weighted else counts
    return {
        "field": field,
        "minimum_score": minimum_score,
        "metric_basis": "inverse-probability weighted uniform_core sample"
        if design_weighted
        else "unweighted reviewed sample counts",
        "raw_counts": dict(counts),
        "design_weighted_estimated_counts": dict(weighted) if design_weighted else None,
        "unsupported_truth_predicted_supported_rate": rate(
            table["unsupported_truth_predicted_supported_false_accept"],
            unsupported_denom,
        ),
        "supported_truth_predicted_unsupported_rate": rate(
            table["supported_truth_predicted_unsupported_false_reject"],
            supported_denom,
        ),
        "supported_truth_quarantined_rate": rate(
            table["supported_truth_quarantined_below_score_floor"]
            + table["supported_truth_quarantined_missing_or_invalid_score"]
            + table["supported_truth_quarantined_unresolved_prediction"],
            supported_denom,
        ),
        "unsupported_truth_quarantined_rate": rate(
            table["unsupported_truth_quarantined_below_score_floor"]
            + table["unsupported_truth_quarantined_missing_or_invalid_score"]
            + table["unsupported_truth_quarantined_unresolved_prediction"],
            unsupported_denom,
        ),
        "ambiguous_truth_count_excluded_from_binary_rates": counts["truth_ambiguous"],
        "note": "Score floors only move records with low scores to quarantine; they do not change the top predicted language. Quarantine counts include all reviewed rows, including mixed/uncertain human labels; those labels are excluded from binary supported-vs-unsupported error rates and reported separately.",
    }


def score_length_diagnostics(
    records: Sequence[Mapping[str, Any]], field: str
) -> list[dict[str, Any]]:
    buckets = (
        (0, 40, "0-40"),
        (41, 100, "41-100"),
        (101, 250, "101-250"),
        (251, 500, "251-500"),
        (501, 1_000, "501-1000"),
        (1_001, 2_000, "1001-2000"),
        (2_001, 10**12, "2001+"),
    )
    diagnostics: list[dict[str, Any]] = []
    for lower, upper, label in buckets:
        pairs = []
        for record in records:
            if _selection_group(record) != "uniform_core_probability_sample":
                continue
            prediction = record.get(f"current_{field}_prediction") or {}
            score = _usable_score(prediction)
            length = prediction.get("input_characters")
            if score is not None and isinstance(length, int) and lower <= length <= upper:
                pairs.append((score, length))
        if not pairs:
            continue
        scores = sorted(score for score, _ in pairs)
        diagnostics.append(
            {
                "length_bucket_chars": label,
                "records": len(scores),
                "score_min": scores[0],
                "score_median": statistics.median(scores),
                "score_max": scores[-1],
                "median_input_characters": statistics.median(length for _, length in pairs),
            }
        )
    return diagnostics


def _evaluator_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "question_prediction": record.get("current_question_prediction"),
            "response_prediction": record.get("current_response_prediction"),
        }
        for record in records
    ]


def _field_prediction_counts(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(
        Counter(
            str((record.get(f"current_{field}_prediction") or {}).get("language") or "<missing>")
            for record in records
        )
    )


def _human_label_counts(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(str(record.get(f"human_{field}_language", "")) for record in records))


def _ai_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_row_id",
            "reviewer_id",
            "review_notes",
            "ai_proposed_question_language",
            "ai_proposed_response_language",
            "ai_proposed_mixed_language_status",
            "ai_proposed_question_genuinely_english",
            "ai_proposed_response_genuinely_english",
            "ai_reviewer_id",
            "ai_review_notes",
            "ai_review_status",
            "ai_basis",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reviewed CSV is missing provenance columns: {sorted(missing)}")
        for row in reader:
            source_id = (row.get("source_row_id") or "").strip()
            if not source_id or source_id in rows:
                raise ValueError(f"Missing or duplicate source_row_id in reviewed CSV: {source_id!r}")
            rows[source_id] = row
    return rows


def reclassify_reviewed_records(
    records: Sequence[Mapping[str, Any]], classifier: Any = langid
) -> list[dict[str, Any]]:
    """Rerun the cleaning pipeline's exact field classifier, preserving prior predictions."""
    rerun: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        # The same audit.classify_field function, default 2,000-character cap,
        # and installed langid module are used by the cleaning pipeline.
        row["current_question_prediction"] = classify_field(
            row.get("question"), classifier=classifier
        )
        row["current_response_prediction"] = classify_field(
            row.get("response"), classifier=classifier
        )
        rerun.append(row)
    return rerun


def _prediction_reproduction(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    same_language = 0
    same_score = 0
    compared_scores = 0
    for record in records:
        old = record.get(f"{field}_prediction") or {}
        current = record.get(f"current_{field}_prediction") or {}
        if old.get("language") == current.get("language"):
            same_language += 1
        old_score = old.get("classifier_score")
        new_score = current.get("classifier_score")
        if isinstance(old_score, (int, float)) and isinstance(new_score, (int, float)):
            compared_scores += 1
            if math.isclose(float(old_score), float(new_score), rel_tol=0, abs_tol=1e-9):
                same_score += 1
    return {
        "same_top_label_count": same_language,
        "same_top_label_fraction": same_language / len(records) if records else None,
        "scores_compared": compared_scores,
        "exactly_same_score_count": same_score,
    }


def _mismatch_cases(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for record in records:
        for field in ("question", "response"):
            human = normalize_label(record.get(f"human_{field}_language"))
            prediction = record.get(f"current_{field}_prediction") or {}
            predicted = normalize_label(prediction.get("language"))
            if human is None or predicted is None or human == predicted:
                continue
            truth_class = _truth_class(human)
            prediction_class = _prediction_class(predicted)
            if truth_class == "unsupported" and prediction_class == "supported":
                failure_type = "unsupported_truth_predicted_english_or_hindi"
            elif truth_class == "supported" and prediction_class == "unsupported":
                failure_type = "english_or_hindi_truth_predicted_unsupported"
            elif truth_class == "ambiguous":
                failure_type = "mixed_or_uncertain_review_label"
            else:
                failure_type = "other_language_label_mismatch"
            text = record.get(field)
            cases.append(
                {
                    "source_row_id": record.get("source_row_id"),
                    "source_shard": record.get("source_shard"),
                    "source_row_number_1_based": record.get("source_row_number_1_based"),
                    "sampling_group": _selection_group(record),
                    "field": field,
                    "human_language": human,
                    "predicted_language": predicted,
                    "failure_type": failure_type,
                    "classifier_score": prediction.get("classifier_score"),
                    "score_is_probability": False,
                    "text_excerpt": text[:MAX_ERROR_PREVIEW_CHARS] if isinstance(text, str) else None,
                }
            )
    priority = {
        "unsupported_truth_predicted_english_or_hindi": 0,
        "english_or_hindi_truth_predicted_unsupported": 1,
        "other_language_label_mismatch": 2,
        "mixed_or_uncertain_review_label": 3,
    }
    return sorted(cases, key=lambda item: (priority[item["failure_type"]], item["source_row_id"], item["field"]))


def build_evaluation(
    records: Sequence[Mapping[str, Any]],
    *,
    classifier: Any = langid,
    minimum_support: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if minimum_support < 1:
        raise ValueError("minimum_support must be at least 1")
    rerun = reclassify_reviewed_records(records, classifier)
    core = [r for r in rerun if _selection_group(r) == "uniform_core_probability_sample"]
    targeted = [r for r in rerun if _selection_group(r) == "targeted_supplemental"]
    if len(core) + len(targeted) != len(rerun):
        raise ValueError("Every reviewed record must belong to exactly one known pilot group")

    group_results: dict[str, Any] = {}
    for name, subset in (
        ("uniform_core_probability_sample", core),
        ("targeted_supplemental", targeted),
    ):
        evaluated = _evaluator_records(subset)
        evaluator_summary, _ = evaluate_records(evaluated, min_support=minimum_support)
        group_results[name] = {
            "record_count": len(subset),
            "human_label_counts": {
                field: _human_label_counts(subset, field)
                for field in ("question", "response")
            },
            "current_prediction_counts": {
                field: _field_prediction_counts(subset, field)
                for field in ("question", "response")
            },
            "field_metrics": evaluator_summary["fields"],
            "note": "The probability-core group supports population-oriented estimates using recorded inclusion probabilities. The targeted group is for failure discovery and edge cases, not population estimation.",
        }

    weighted_core_metrics = {}
    for field in ("question", "response"):
        weighted_evaluated = _evaluator_records(core)
        # The existing evaluator reports the sample confusion matrix. Add a
        # design-weighted matrix using the two recorded sampling probabilities.
        labels = sorted(
            {
                str(normalize_label(row.get(f"human_{field}_language")))
                for row in core
                if normalize_label(row.get(f"human_{field}_language")) is not None
            }
            | {
                str(normalize_label((row.get(f"current_{field}_prediction") or {}).get("language")))
                for row in core
                if normalize_label((row.get(f"current_{field}_prediction") or {}).get("language")) is not None
            }
        )
        matrix: dict[str, dict[str, float]] = {label: {p: 0.0 for p in labels} for label in labels}
        weight_sum = 0.0
        correct_weight = 0.0
        for row in core:
            human = normalize_label(row.get(f"human_{field}_language"))
            predicted = normalize_label((row.get(f"current_{field}_prediction") or {}).get("language"))
            if human is None or predicted is None:
                continue
            weight = _design_weight(row)
            matrix[human][predicted] += weight
            weight_sum += weight
            if human == predicted:
                correct_weight += weight
        weighted_per_language = []
        for label in labels:
            support = sum(matrix[label].values())
            predicted_count = sum(matrix[actual][label] for actual in labels)
            true_positive = matrix[label][label]
            precision = true_positive / predicted_count if predicted_count else None
            recall = true_positive / support if support else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            )
            weighted_per_language.append(
                {
                    "language": label,
                    "estimated_human_support": support,
                    "estimated_predicted_count": predicted_count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        weighted_core_metrics[field] = {
            "weighted_confusion_matrix_estimated_population_counts": matrix,
            "weighted_accuracy_ratio_estimate": correct_weight / weight_sum if weight_sum else None,
            "weighted_per_language_metrics": weighted_per_language,
            "sum_of_weights_on_compared_records": weight_sum,
            "interpretation": "Ratio estimates from the 400-row probability-core sample; targeted rows are excluded.",
        }

    score_analysis: dict[str, Any] = {}
    for field in ("question", "response"):
        floors = candidate_score_floors(rerun, field)
        score_analysis[field] = {
            "candidate_floors_source": "Observed current langid scores in the uniform_core probability subset; quantiles are diagnostic only and are not selected to optimize agreement.",
            "candidate_floors": [
                {
                    **candidate,
                    "uniform_core": analyze_score_floor(
                        core,
                        field,
                        candidate["minimum_score"],
                        design_weighted=True,
                    ),
                    "targeted_supplemental": analyze_score_floor(
                        targeted,
                        field,
                        candidate["minimum_score"],
                    ),
                }
                for candidate in floors
            ],
            "score_by_core_text_length": score_length_diagnostics(core, field),
        }

    label_status_counts = {
        "mixed_language_status": dict(
            Counter(str(r.get("human_mixed_language_status", "")) for r in rerun)
        ),
        "question_genuinely_english": dict(
            Counter(str(r.get("question_genuinely_english", "")) for r in rerun)
        ),
        "response_genuinely_english": dict(
            Counter(str(r.get("response_genuinely_english", "")) for r in rerun)
        ),
    }
    ambiguous_label_analysis = {}
    for field in ("question", "response"):
        ambiguous = [
            row
            for row in rerun
            if normalize_label(row.get(f"human_{field}_language")) in {"mixed", "uncertain"}
        ]
        ambiguous_label_analysis[field] = {
            "record_count": len(ambiguous),
            "human_language_labels": dict(
                Counter(str(normalize_label(row.get(f"human_{field}_language"))) for row in ambiguous)
            ),
            "predicted_language_counts": _field_prediction_counts(ambiguous, field),
            "examples": [
                {
                    "source_row_id": row.get("source_row_id"),
                    "sampling_group": _selection_group(row),
                    "human_language": normalize_label(row.get(f"human_{field}_language")),
                    "predicted_language": (row.get(f"current_{field}_prediction") or {}).get("language"),
                    "text_excerpt": (row.get(field) or "")[:MAX_ERROR_PREVIEW_CHARS]
                    if isinstance(row.get(field), str)
                    else None,
                }
                for row in ambiguous[:20]
            ],
        }
    score_semantics = {
        "langid_version": language_classifier_version(),
        "normalization_enabled": bool(langid_impl.NORM_PROBS),
        "classify_selects": "argmax of the classifier's returned score vector",
        "rank_order": "descending score (the highest numeric score is ranked first)",
        "score_kind": "unnormalized langid class log-score in this installed configuration",
        "score_is_probability": False,
        "length_warning": "The score is an additive score over observed text features and its magnitude varies with text length; it is not a calibrated probability or directly comparable confidence across arbitrary lengths.",
    }
    errors = _mismatch_cases(rerun)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewed_csv": str(DEFAULT_REVIEWED_CSV),
        "reviewed_jsonl": str(DEFAULT_REVIEWED_JSONL),
        "review_provenance": "AI-assisted labels were subsequently reviewed and confirmed by the user. This is one user review, not independent annotation by multiple human reviewers.",
        "reviewed_record_count": len(rerun),
        "sampling_design_counts": {
            "uniform_core_probability_sample": len(core),
            "targeted_supplemental": len(targeted),
            "combined_accuracy_estimate_reported": False,
        },
        "classifier_reexecution": {
            "name": "langid",
            "version": language_classifier_version(),
            "text_limit_per_field_chars": CLASSIFIER_TEXT_LIMIT,
            "fields_classified_separately": True,
            "context_used": False,
            "current_predictions_stored_separately": True,
            "original_prediction_reproduction": {
                field: _prediction_reproduction(rerun, field)
                for field in ("question", "response")
            },
        },
        "score_semantics": score_semantics,
        "label_status_counts": label_status_counts,
        "mixed_and_uncertain_field_analysis": ambiguous_label_analysis,
        "results_by_sampling_group": group_results,
        "design_weighted_core_metrics": weighted_core_metrics,
        "score_floor_analysis": score_analysis,
        "mismatch_case_count": len(errors),
        "mismatch_counts_by_type": dict(Counter(error["failure_type"] for error in errors)),
        "limitations": [
            "The core metrics describe a 400-record stratified probability sample; design-weighted ratio estimates still have substantial sampling uncertainty.",
            "The 100 targeted supplemental records are deliberately enriched for unusual predictions and text patterns; their metrics are not population estimates.",
            "Some unsupported languages have very few or no reviewed examples, so performance for every unsupported language is not established.",
            "Mixed and uncertain labels are not binary threshold ground truth. Mixed-language constituent validation remains separate from score-floor analysis.",
            "A raw score floor based on this pilot is exploratory and cannot by itself authorize full-corpus PASS/REJECT outcomes.",
        ],
        "next_validation_recommendation": {
            "initial_additional_targeted_pairs": 100,
            "strata": [
                "Hindi predictions and reviewed Hindi fields",
                "unsupported-language predictions and reviewed unsupported fields",
                "Romanized Hindi",
                "mixed and uncertain fields",
                "short and long text",
            ],
            "field_level_support_goal": "At least 30 usable reviewed examples for Hindi and unsupported-language outcomes in each field where feasible; extend adaptively if not reached.",
            "design_note": "Record inclusion probabilities. This targeted increment is for failure discovery and class support, not a population estimate, coverage of every unsupported language, or approval of score floors. Maintain a separate probability sample for population-oriented estimates.",
        },
    }
    return summary, errors, rerun


def attach_ai_provenance(
    records: Sequence[Mapping[str, Any]], csv_path: Path
) -> list[dict[str, Any]]:
    csv_rows = _ai_csv_rows(csv_path)
    record_ids = {str(record.get("source_row_id")) for record in records}
    if record_ids != set(csv_rows):
        raise ValueError("Reviewed CSV and JSONL source_row_id sets do not match")
    output: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        csv_row = csv_rows[str(row["source_row_id"])]
        row["ai_proposed_labels"] = {
            key.removeprefix("ai_proposed_"): csv_row[key]
            for key in (
                "ai_proposed_question_language",
                "ai_proposed_response_language",
                "ai_proposed_mixed_language_status",
                "ai_proposed_question_genuinely_english",
                "ai_proposed_response_genuinely_english",
            )
        }
        row["ai_review_provenance"] = {
            key: csv_row[key]
            for key in ("ai_reviewer_id", "ai_review_notes", "ai_review_status", "ai_basis")
        }
        row["user_review_provenance"] = {
            "reviewer_id": csv_row["reviewer_id"],
            "review_notes": csv_row["review_notes"],
            "status": "user_confirmed_ai_assisted_draft",
            "independent_multiple_reviewer_annotation": False,
        }
        output.append(row)
    return output


def write_outputs(
    summary: dict[str, Any],
    errors: Sequence[Mapping[str, Any]],
    rerun_records: Sequence[Mapping[str, Any]],
    csv_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "language_classifier_evaluation_user_reviewed_v1_refined_v3.json",
        "report": output_dir / "language_classifier_evaluation_user_reviewed_v1_refined_v3.md",
        "errors": output_dir / "language_classifier_evaluation_errors_user_reviewed_v1_refined_v3.jsonl",
        "evaluated_records": output_dir / "language_review_pilot_user_reviewed_v1_evaluated_refined_v3.jsonl",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite evaluation outputs: {existing}")

    summary["output_files"] = {key: path.name for key, path in paths.items()}
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with paths["errors"].open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False, default=str) + "\n")

    enriched_records = attach_ai_provenance(rerun_records, csv_path)
    with paths["evaluated_records"].open("w", encoding="utf-8") as handle:
        for row in enriched_records:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    paths["report"].write_text(render_markdown_report(summary, errors), encoding="utf-8")
    return paths


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _fmt_count(value: int | float, *, estimated: bool = False) -> str:
    if estimated:
        return f"{float(value):,.1f}"
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.1f}"


def _supported_error_display(
    counts: Mapping[str, Any],
    numerator_key: str,
    denominator_key: str,
    rate: float | None,
    *,
    estimated: bool,
) -> str:
    numerator = counts.get(numerator_key, 0)
    denominator = counts.get(denominator_key, 0)
    if not denominator:
        return "n/a (0/0 support)"
    if estimated:
        return (
            f"{float(numerator):,.1f}/{float(denominator):,.1f} est.; "
            f"{_fmt_percent(rate)}"
        )
    return f"{int(numerator)}/{int(denominator)}; {_fmt_percent(rate)}"


def _confidence_quarantine_count(counts: Mapping[str, Any]) -> float:
    """Count binary-labeled rows quarantined for score or prediction uncertainty."""
    return sum(
        float(counts.get(key, 0))
        for key in (
            "quarantined_below_score_floor",
            "quarantined_missing_or_invalid_score",
            "quarantined_unresolved_prediction",
        )
    )


def _nonzero_confusion_cells(metric: Mapping[str, Any]) -> str:
    confusion = metric.get("confusion_matrix") or {}
    labels = confusion.get("labels", [])
    counts = confusion.get("counts", {})
    cells = [
        f"{actual}→{predicted}: {counts.get(actual, {}).get(predicted, 0)}"
        for actual in labels
        for predicted in labels
        if counts.get(actual, {}).get(predicted, 0)
    ]
    return "; ".join(cells) if cells else "none"


def render_markdown_report(
    summary: Mapping[str, Any], errors: Sequence[Mapping[str, Any]]
) -> str:
    lines = [
        "# Language classifier evaluation: user-reviewed pilot",
        "",
        f"- Reviewed records: **{summary['reviewed_record_count']:,}**",
        f"- Sampling groups: **{summary['sampling_design_counts']['uniform_core_probability_sample']:,} core probability**, **{summary['sampling_design_counts']['targeted_supplemental']:,} targeted**",
        "- The combined 500 rows are not reported as a simple-random-sample accuracy estimate.",
        f"- Review provenance: {summary['review_provenance']}",
        f"- Classifier: `langid` {summary['classifier_reexecution']['version']}; per-field limit {summary['classifier_reexecution']['text_limit_per_field_chars']:,} chars.",
        "",
        "## Score semantics",
        "",
        f"- Installed `NORM_PROBS`: `{summary['score_semantics']['normalization_enabled']}`.",
        f"- {summary['score_semantics']['rank_order']}; {summary['score_semantics']['score_kind']}.",
        f"- {summary['score_semantics']['length_warning']}",
        "- Raw scores are not reported as probabilities.",
        "",
        "## Metrics by sampling group",
        "",
    ]
    for group_name, group in summary["results_by_sampling_group"].items():
        lines.extend([f"### {group_name.replace('_', ' ').title()} ({group['record_count']:,} records)", ""])
        for field in ("question", "response"):
            metric = group["field_metrics"][field]
            lines.extend(
                [
                    f"**{field.title()}** — evaluated {metric['n_compared']:,}; accuracy {_fmt(metric['accuracy'])}.",
                    f"Confusion matrix (human→predicted; nonzero cells): {_nonzero_confusion_cells(metric)}.",
                    "",
                    "Per-language metrics (per-language sufficiency is flagged in JSON):",
                    "",
                    "| Label | Human support | Predicted | Precision | Recall | F1 | Sufficient support |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for item in metric["per_language"]:
                lines.append(
                    f"| {item['language']} | {item['human_support']} | {item['predicted_count']} | {_fmt(item['precision'])} | {_fmt(item['recall'])} | {_fmt(item['f1'])} | {item['reported_as_sufficient_support']} |"
                )
            lines.extend(
                [
                    "",
                    f"Human labels: `{group['human_label_counts'][field]}`.",
                    f"Current predictions: `{group['current_prediction_counts'][field]}`.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Candidate score floors",
            "",
            "Floors are observed-score quantiles from the core probability sample; they were not optimized against predicted labels. Binary rates exclude human `mixed` and `uncertain` labels, which are analyzed separately. Confidence-quarantine counts include all reviewed rows, including mixed/uncertain labels, and include below-floor, missing/invalid-score, and unresolved-prediction cases. Core entries are inverse-probability weighted estimates; targeted entries are raw edge-case counts.",
            "",
        ]
    )
    for field, analysis in summary["score_floor_analysis"].items():
        lines.extend(
            [
                f"### {field.title()}",
                "",
                "| Candidate | Raw score floor | Core false accept (est. n/support; rate) | Core false reject (est. n/support; rate) | Core confidence quarantine (est. n) | Targeted false accept (n/support; rate) | Targeted false reject (n/support; rate) | Targeted confidence quarantine (n) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in analysis["candidate_floors"]:
            core = candidate["uniform_core"]
            target = candidate["targeted_supplemental"]
            core_counts = core["design_weighted_estimated_counts"]
            target_counts = target["raw_counts"]
            lines.append(
                f"| {candidate['candidate']} | {_fmt(candidate['minimum_score'])} | "
                f"{_supported_error_display(core_counts, 'unsupported_truth_predicted_supported_false_accept', 'truth_unsupported', core['unsupported_truth_predicted_supported_rate'], estimated=True)} | "
                f"{_supported_error_display(core_counts, 'supported_truth_predicted_unsupported_false_reject', 'truth_supported', core['supported_truth_predicted_unsupported_rate'], estimated=True)} | "
                f"{_fmt_count(_confidence_quarantine_count(core_counts), estimated=True)} | "
                f"{_supported_error_display(target_counts, 'unsupported_truth_predicted_supported_false_accept', 'truth_unsupported', target['unsupported_truth_predicted_supported_rate'], estimated=False)} | "
                f"{_supported_error_display(target_counts, 'supported_truth_predicted_unsupported_false_reject', 'truth_supported', target['supported_truth_predicted_unsupported_rate'], estimated=False)} | "
                f"{_fmt_count(_confidence_quarantine_count(target_counts))} |"
            )
        lines.extend(["", "Core score ranges by field length:", ""])
        lines.append("| Length chars | n | Min score | Median score | Max score | Median chars |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in analysis["score_by_core_text_length"]:
            lines.append(
                f"| {row['length_bucket_chars']} | {row['records']} | {_fmt(row['score_min'])} | {_fmt(row['score_median'])} | {_fmt(row['score_max'])} | {_fmt(row['median_input_characters'])} |"
            )
        lines.append("")

    lines.extend(["## Important mismatches", ""])
    if not errors:
        lines.append("No classifier/label mismatches were found.")
    else:
        lines.extend(
            [
                "All mismatches are preserved in the JSONL error file; examples below are short excerpts.",
                "",
                "| Source row | Group | Field | Reviewed | Predicted | Score | Failure | Excerpt |",
                "|---|---|---|---|---|---:|---|---|",
            ]
        )
        for error in errors[:15]:
            excerpt = (error.get("text_excerpt") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{error['source_row_id']}` | {error['sampling_group']} | {error['field']} | {error['human_language']} | {error['predicted_language']} | {_fmt(error['classifier_score'])} | {error['failure_type']} | {excerpt} |"
            )
    lines.extend(
        [
            "",
            "## Recommendation and limitations",
            "",
            "No production score floor is recommended from this pilot. Candidate floors are exploratory only. The sample has limited support for many unsupported languages, the targeted subset is not representative, and the raw score magnitude varies with text length. Mixed-language constituent validation remains a separate human-review requirement. Keep the language gate disabled pending a larger stratified review and explicit approval.",
            "",
            "**Next review:** add an initial targeted increment of about 100 pairs covering Hindi and unsupported-language cases by field, Romanized Hindi, mixed/uncertain text, and short/long fields. Continue adaptively toward at least 30 usable reviewed examples of Hindi and unsupported-language outcomes per field where feasible, recording inclusion probabilities. Keep a separate probability sample for population estimates. This follow-up is not coverage of every unsupported language and does not approve score floors.",
            "",
            "See the JSON summary for confusion matrices, weighted core estimates, review-label counts, exact reproduction statistics, and limitations.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a reviewed pilot and inspect exploratory langid score floors."
    )
    parser.add_argument("--reviewed-jsonl", type=Path, default=DEFAULT_REVIEWED_JSONL)
    parser.add_argument("--reviewed-csv", type=Path, default=DEFAULT_REVIEWED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_DIR / "language_review_template.json")
    parser.add_argument("--minimum-support", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.reviewed_jsonl)
    csv_rows = _ai_csv_rows(args.reviewed_csv)
    review_validation = validate_reviewed_labels(csv_rows, records, args.template)
    summary, errors, rerun = build_evaluation(
        records, classifier=langid, minimum_support=args.minimum_support
    )
    summary["reviewed_csv"] = str(args.reviewed_csv)
    summary["reviewed_jsonl"] = str(args.reviewed_jsonl)
    summary["review_validation"] = review_validation
    paths = write_outputs(summary, errors, rerun, args.reviewed_csv, args.output_dir)
    print(
        f"Evaluated {len(rerun)} rows; mismatches={len(errors)}; wrote "
        f"{paths['summary']} and {paths['report']}",
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover - CLI exercised by smoke run
    main()
