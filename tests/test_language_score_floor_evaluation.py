from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.data_pipeline.audit.evaluate_reviewed_language_pilot import (
    _confidence_quarantine_count,
    _supported_error_display,
    analyze_score_floor,
    build_evaluation,
    candidate_score_floors,
    reclassify_reviewed_records,
    render_markdown_report,
    validate_reviewed_labels,
)


def prediction(language: str, score: float) -> dict[str, Any]:
    return {
        "status": "classified",
        "language": language,
        "classifier_score": score,
        "score_is_probability": False,
        "input_characters": 50,
        "classified_characters": 50,
        "input_truncated": False,
    }


def make_record(
    *,
    row_id: str,
    group: str,
    question_truth: str = "en",
    response_truth: str = "en",
    question_prediction: dict[str, Any] | None = None,
    response_prediction: dict[str, Any] | None = None,
    score: float = -1.0,
) -> dict[str, Any]:
    if question_prediction is None:
        question_prediction = prediction("en", score)
    if response_prediction is None:
        response_prediction = prediction("en", score)
    is_core = group == "uniform_core_probability_sample"
    return {
        "source_row_id": row_id,
        "source_shard": "train-00000-of-00033.parquet",
        "source_row_number_1_based": 1,
        "sampling_basis": ["uniform_core"] if is_core else ["supplemental_targeted"],
        "uniform_core_inclusion_probability": 0.1 if is_core else None,
        "pilot_selection": {
            "pilot_conditional_inclusion_probability": 0.5 if is_core else None,
        },
        "question": "A short question?",
        "response": "A short response.",
        "question_prediction": prediction("en", score),
        "response_prediction": prediction("en", score),
        "current_question_prediction": question_prediction,
        "current_response_prediction": response_prediction,
        "human_question_language": question_truth,
        "human_response_language": response_truth,
        "human_mixed_language_status": "none",
        "question_genuinely_english": "yes" if question_truth == "en" else "no",
        "response_genuinely_english": "yes" if response_truth == "en" else "no",
    }


class FakeClassifier:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def classify(self, text: str) -> tuple[str, float]:
        self.seen.append(text)
        return "en", -3.0


class ScoreFloorAnalysisTests(unittest.TestCase):
    def test_false_accept_rate_shows_unsupported_class_support(self):
        self.assertEqual(
            _supported_error_display({}, "false_accept", "truth_unsupported", None, estimated=False),
            "n/a (0/0 support)",
        )
        self.assertEqual(
            _supported_error_display(
                {"false_accept": 2, "truth_unsupported": 10},
                "false_accept",
                "truth_unsupported",
                0.2,
                estimated=False,
            ),
            "2/10; 20.0%",
        )

    def test_confidence_quarantine_includes_missing_unresolved_and_below_floor(self):
        self.assertEqual(
            _confidence_quarantine_count(
                {
                    "quarantined_below_score_floor": 2,
                    "quarantined_missing_or_invalid_score": 1,
                    "quarantined_unresolved_prediction": 3,
                }
            ),
            6,
        )

    def test_report_includes_confusion_cells_and_marks_zero_support_as_not_estimable(self):
        rows = [
            make_record(row_id="core", group="uniform_core_probability_sample"),
            make_record(row_id="target", group="targeted_supplemental", question_truth="ur"),
        ]
        summary, errors, _ = build_evaluation(rows, classifier=FakeClassifier())
        report = render_markdown_report(summary, errors)
        self.assertIn("Confusion matrix (human→predicted; nonzero cells)", report)
        self.assertIn("en→en: 1", report)
        self.assertIn("n/a (0/0 support)", report)

    def test_candidate_floors_use_observed_core_scores_not_labels_or_targeted_scores(self):
        rows = [
            make_record(row_id="core-1", group="uniform_core_probability_sample", score=-10),
            make_record(row_id="core-2", group="uniform_core_probability_sample", score=-5),
            make_record(row_id="core-3", group="uniform_core_probability_sample", score=-1),
            make_record(row_id="target", group="targeted_supplemental", score=999),
        ]
        floors = candidate_score_floors(rows, "question", quantiles=(0, 0.5, 1))
        self.assertEqual([item["minimum_score"] for item in floors], [None, -10.0, -5.0, -1.0])
        rows[0]["human_question_language"] = "ur"
        self.assertEqual(candidate_score_floors(rows, "question", quantiles=(0, 0.5, 1)), floors)

    def test_floor_counts_false_accept_false_reject_and_quarantine_separately(self):
        rows = [
            make_record(row_id="supported-ok", group="uniform_core_probability_sample", question_truth="en", score=-1),
            make_record(row_id="unsupported-as-en", group="uniform_core_probability_sample", question_truth="ur", score=-0.5),
            make_record(
                row_id="english-as-ur",
                group="uniform_core_probability_sample",
                question_truth="en",
                question_prediction=prediction("ur", -0.25),
            ),
            make_record(
                row_id="low-score",
                group="uniform_core_probability_sample",
                question_truth="hi",
                question_prediction=prediction("hi", -4),
            ),
            make_record(
                row_id="mixed",
                group="uniform_core_probability_sample",
                question_truth="mixed",
                question_prediction=prediction("en", -0.1),
            ),
        ]
        result = analyze_score_floor(rows, "question", -2.0)
        counts = result["raw_counts"]
        self.assertEqual(counts["unsupported_truth_predicted_supported_false_accept"], 1)
        self.assertEqual(counts["supported_truth_predicted_unsupported_false_reject"], 1)
        self.assertEqual(counts["quarantined_below_score_floor"], 1)
        self.assertEqual(result["ambiguous_truth_count_excluded_from_binary_rates"], 1)

    def test_inverse_probability_weight_multiplies_core_and_pilot_sampling(self):
        rows = [make_record(row_id="core", group="uniform_core_probability_sample")]
        result = analyze_score_floor(rows, "question", None, design_weighted=True)
        self.assertEqual(
            result["design_weighted_estimated_counts"]["supported_truth_retained_as_supported"],
            20,
        )
        self.assertEqual(result["metric_basis"], "inverse-probability weighted uniform_core sample")

    def test_mixed_and_uncertain_labels_are_not_forced_into_binary_truth(self):
        rows = [
            make_record(
                row_id="mixed",
                group="uniform_core_probability_sample",
                question_truth="mixed",
            ),
            make_record(
                row_id="uncertain",
                group="uniform_core_probability_sample",
                question_truth="uncertain",
            ),
        ]
        result = analyze_score_floor(rows, "question", None)
        self.assertEqual(result["raw_counts"]["truth_ambiguous"], 2)
        self.assertEqual(result["ambiguous_truth_count_excluded_from_binary_rates"], 2)
        self.assertIsNone(result["unsupported_truth_predicted_supported_rate"])

    def test_mixed_and_uncertain_rows_still_count_toward_confidence_quarantine(self):
        row = make_record(
            row_id="mixed-low-score",
            group="uniform_core_probability_sample",
            question_truth="mixed",
            question_prediction=prediction("en", -4.0),
        )
        result = analyze_score_floor([row], "question", -2.0)
        self.assertEqual(result["raw_counts"]["truth_ambiguous"], 1)
        self.assertEqual(result["raw_counts"]["quarantined_below_score_floor"], 1)
        self.assertEqual(result["ambiguous_truth_count_excluded_from_binary_rates"], 1)

    def test_reclassification_keeps_original_predictions_and_classifies_fields_separately(self):
        row = make_record(
            row_id="one",
            group="uniform_core_probability_sample",
            question_prediction=prediction("hi", -30),
            response_prediction=prediction("ur", -40),
        )
        row["question_prediction"] = prediction("hi", -30)
        row["response_prediction"] = prediction("ur", -40)
        classifier = FakeClassifier()
        rerun = reclassify_reviewed_records([row], classifier)
        self.assertEqual(classifier.seen, ["A short question?", "A short response."])
        self.assertEqual(rerun[0]["question_prediction"]["language"], "hi")
        self.assertEqual(rerun[0]["response_prediction"]["language"], "ur")
        self.assertEqual(rerun[0]["current_question_prediction"]["language"], "en")
        self.assertEqual(rerun[0]["current_response_prediction"]["language"], "en")
        self.assertFalse(rerun[0]["current_question_prediction"]["score_is_probability"])

    def test_evaluation_reports_core_and_targeted_groups_separately(self):
        rows = [
            make_record(row_id="core-1", group="uniform_core_probability_sample"),
            make_record(row_id="core-2", group="uniform_core_probability_sample", question_truth="hi"),
            make_record(row_id="target-1", group="targeted_supplemental", question_truth="zh"),
        ]
        summary, errors, rerun = build_evaluation(rows, classifier=FakeClassifier())
        groups = summary["results_by_sampling_group"]
        self.assertEqual(groups["uniform_core_probability_sample"]["record_count"], 2)
        self.assertEqual(groups["targeted_supplemental"]["record_count"], 1)
        self.assertFalse(summary["sampling_design_counts"]["combined_accuracy_estimate_reported"])
        self.assertEqual(len(rerun), 3)
        self.assertTrue(errors)
        self.assertIn("weighted_per_language_metrics", summary["design_weighted_core_metrics"]["question"])

    def test_review_label_validation_reports_missing_invalid_and_inconsistent_values(self):
        record = make_record(row_id="rev:shard:1", group="uniform_core_probability_sample")
        csv_row = {
            "human_question_language": "mixed",
            "human_response_language": "en",
            "human_mixed_language_status": "none",
            "question_genuinely_english": "yes",
            "response_genuinely_english": "yes",
            "reviewer_id": "User",
            "review_notes": "Reviewed",
        }
        template = {
            "fields": {
                "human_mixed_language_status": {"allowed_values": ["none", "question", "response", "both", "uncertain"]},
                "question_genuinely_english": {"allowed_values": ["yes", "no", "partial", "uncertain"]},
                "response_genuinely_english": {"allowed_values": ["yes", "no", "partial", "uncertain"]},
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            template_path = Path(temporary) / "template.json"
            import json
            template_path.write_text(json.dumps(template))
            result = validate_reviewed_labels({"rev:shard:1": csv_row}, [record], template_path)
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["semantic_consistency_issue_count"], 2)
        self.assertEqual(result["missing_field_counts"], {})


if __name__ == "__main__":
    unittest.main()
