"""Compare completed human language labels with stored langid predictions.

The script reports metrics on the reviewed records only. It does not set a
language threshold or authorize any dataset filtering.

Example:
  uv run python scripts/data_pipeline/audit/evaluate_language_review.py \
    data/reports/language-validation-v1/language_review.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

LANGUAGE_NAME_ALIASES = {
    "english": "en",
    "hindi": "hi",
    "mandarin": "zh",
    "chinese": "zh",
    "latin": "la",
    "mixed-language": "mixed",
    "mixed language": "mixed",
}
VALID_ENGLISHNESS = {"yes", "no", "partial", "uncertain"}


def normalize_label(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    label = value.strip().lower().replace("_", "-")
    return LANGUAGE_NAME_ALIASES.get(label, label)


def _confusion_matrix(
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    labels = sorted({gold for gold, _ in pairs} | {pred for _, pred in pairs})
    counts: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in labels} for gold in labels
    }
    for gold, prediction in pairs:
        counts[gold][prediction] += 1
    return {"labels": labels, "counts": counts}


def _per_language_metrics(
    pairs: list[tuple[str, str]], min_support: int
) -> list[dict[str, Any]]:
    support: Counter[str] = Counter(gold for gold, _ in pairs)
    predicted: Counter[str] = Counter(pred for _, pred in pairs)
    true_positive: Counter[str] = Counter(
        gold for gold, pred in pairs if gold == pred
    )
    labels = sorted(set(support) | set(predicted))
    metrics: list[dict[str, Any]] = []
    for label in labels:
        actual_count = support[label]
        predicted_count = predicted[label]
        precision = (
            true_positive[label] / predicted_count if predicted_count else None
        )
        recall = true_positive[label] / actual_count if actual_count else None
        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        metrics.append(
            {
                "language": label,
                "human_support": actual_count,
                "predicted_count": predicted_count,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "precision_has_minimum_support": predicted_count >= min_support,
                "recall_has_minimum_support": actual_count >= min_support,
                "reported_as_sufficient_support": actual_count >= min_support
                and predicted_count >= min_support,
                "minimum_support_for_reporting": min_support,
            }
        )
    return metrics


def _englishness_metrics(
    records: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    human_key = f"{field}_genuinely_english"
    prediction_key = f"{field}_prediction"
    pairs: list[tuple[str, str]] = []
    skipped: Counter[str] = Counter()
    for record in records:
        gold = normalize_label(record.get(human_key))
        prediction = record.get(prediction_key) or {}
        predicted_language = normalize_label(prediction.get("language"))
        if gold not in VALID_ENGLISHNESS:
            skipped["missing_or_invalid_human_label"] += 1
            continue
        if not predicted_language:
            skipped["missing_prediction"] += 1
            continue
        if gold in {"partial", "uncertain"}:
            skipped[f"human_{gold}"] += 1
            continue
        predicted_english = "yes" if predicted_language == "en" else "no"
        pairs.append((gold, predicted_english))
    correct = sum(gold == pred for gold, pred in pairs)
    return {
        "n_compared": len(pairs),
        "accuracy": correct / len(pairs) if pairs else None,
        "confusion_matrix": _confusion_matrix(pairs),
        "skipped_counts": dict(skipped),
        "note": "Uses langid's top predicted label only; this is a sample comparison, not a calibrated threshold or an approved filter.",
    }


def _evaluate_subset(
    records: list[dict[str, Any]], min_support: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    field_results: dict[str, Any] = {}
    error_cases: list[dict[str, Any]] = []
    for field in ("question", "response"):
        human_key = f"human_{field}_language"
        prediction_key = f"{field}_prediction"
        pairs: list[tuple[str, str]] = []
        skipped: Counter[str] = Counter()
        labeled_records = 0
        for record in records:
            gold = normalize_label(record.get(human_key))
            prediction = record.get(prediction_key) or {}
            predicted_language = normalize_label(prediction.get("language"))
            if gold is None:
                skipped["missing_human_label"] += 1
                continue
            labeled_records += 1
            if not predicted_language:
                skipped["missing_or_failed_prediction"] += 1
                continue
            pairs.append((gold, predicted_language))
            if gold != predicted_language:
                error_cases.append(
                    {
                        "source_row_id": record.get("source_row_id"),
                        "source_shard": record.get("source_shard"),
                        "source_row_number_1_based": record.get(
                            "source_row_number_1_based"
                        ),
                        "field": field,
                        "human_language": gold,
                        "predicted_language": predicted_language,
                        "classifier_score": prediction.get("classifier_score"),
                        "sampling_basis": record.get("sampling_basis", []),
                        "uniform_core_inclusion_probability": record.get(
                            "uniform_core_inclusion_probability"
                        ),
                        "text": record.get(field),
                    }
                )
        correct = sum(gold == pred for gold, pred in pairs)
        field_results[field] = {
            "human_labeled_records": labeled_records,
            "n_compared": len(pairs),
            "accuracy": correct / len(pairs) if pairs else None,
            "confusion_matrix": _confusion_matrix(pairs),
            "per_language": _per_language_metrics(pairs, min_support),
            "skipped_counts": dict(skipped),
            "genuinely_english_comparison": _englishness_metrics(records, field),
        }

    mixed_status_counts = Counter(
        str(record.get("human_mixed_language_status", "")).strip().lower()
        for record in records
        if str(record.get("human_mixed_language_status", "")).strip()
    )
    englishness_review_counts = {
        field: dict(
            Counter(
                str(record.get(f"{field}_genuinely_english", ""))
                .strip()
                .lower()
                for record in records
                if str(record.get(f"{field}_genuinely_english", "")).strip()
            )
        )
        for field in ("question", "response")
    }
    summary = {
        "reviewed_records": len(records),
        "minimum_support_for_per_language_reporting": min_support,
        "fields": field_results,
        "human_mixed_language_status_counts": dict(mixed_status_counts),
        "human_genuinely_english_label_counts": englishness_review_counts,
        "error_case_count": len(error_cases),
        "interpretation": [
            "Metrics describe only records with completed human labels in this review file.",
            "A mixed or uncertain human language label is compared as supplied; langid still emits one predicted label.",
            "Per-language metrics below minimum support are included but marked as insufficient support.",
            "These results do not set a classifier-score threshold and do not authorize language-based cleaning or rejection.",
            "These metrics are for this reviewed subset. They are not automatically population estimates.",
        ],
    }
    return summary, error_cases


def evaluate_records(
    records: list[dict[str, Any]], min_support: int = 5
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if min_support < 1:
        raise ValueError("min_support must be at least 1")

    summary, error_cases = _evaluate_subset(records, min_support)
    core_records = [
        record
        for record in records
        if "uniform_core" in record.get("sampling_basis", [])
    ]
    targeted_records = [
        record
        for record in records
        if "supplemental_targeted" in record.get("sampling_basis", [])
    ]
    core_summary, _ = _evaluate_subset(core_records, min_support)
    targeted_summary, _ = _evaluate_subset(targeted_records, min_support)
    summary["sampling_design_subsets"] = {
        "core_stratified_probability_sample": core_summary,
        "supplemental_targeted_review_sample": targeted_summary,
        "note": "The core sample is kept separate from targeted cases. Targeted-only metrics are for error discovery, not corpus estimation. Any probability-oriented estimate must respect the recorded per-shard inclusion probabilities.",
    }
    return summary, error_cases


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            records.append(item)
    return records


def write_evaluation_outputs(
    input_path: Path,
    summary: dict[str, Any],
    errors: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    destination = output_dir or input_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "language_evaluation_summary.json"
    errors_path = destination / "language_review_errors.jsonl"
    summary["input_review_file"] = str(input_path)
    summary["output_files"] = {
        "summary": summary_path.name,
        "error_cases": errors_path.name,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with errors_path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False, default=str) + "\n")
    return summary_path, errors_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate completed human language labels against langid predictions."
    )
    parser.add_argument("review_file", type=Path, help="Completed language_review.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--min-support",
        type=int,
        default=5,
        help="Minimum human or predicted count for marking per-language metrics sufficiently supported; not a classifier threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.review_file)
    summary, errors = evaluate_records(records, min_support=args.min_support)
    summary_path, errors_path = write_evaluation_outputs(
        args.review_file, summary, errors, args.output_dir
    )
    print(f"Wrote {summary_path} and {errors_path}", flush=True)


if __name__ == "__main__":  # pragma: no cover - covered by API tests
    main()
