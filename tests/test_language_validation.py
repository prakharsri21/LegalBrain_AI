import json
import tempfile
import unittest
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_pipeline.audit.evaluate_language_review import evaluate_records
from scripts.data_pipeline.audit.validate_language import (
    classify_field,
    classify_record_fields,
    collect_validation_sample,
    heuristic_strata,
    write_validation_outputs,
)


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def classify(self, text):
        self.calls.append(text)
        if any("DEVANAGARI" in unicodedata.name(ch, "") for ch in text):
            return "hi", 3.25
        return "en", -8.5


def write_shard(path: Path, records):
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


class LanguageValidationTests(unittest.TestCase):
    def make_records(self, count, shard_number):
        records = []
        for index in range(count):
            question = f"What rule applies in matter {shard_number}-{index}?"
            response = f"The answer is English text for record {shard_number}-{index}."
            if index % 7 == 0:
                question += " Under Section 42 of the Act."
            if index % 9 == 0:
                response = f'“Quoted answer” for party {shard_number}-{index}.'
            if index % 11 == 0:
                response = "न्यायालय ने अपील खारिज की।"
            if index % 13 == 0:
                question = "Who?"
            if index % 17 == 0:
                response = "A replacement � character and\x05 control."
            records.append(
                {
                    "context": f"Provenance context {shard_number}-{index}",
                    "question": question,
                    "response": response,
                }
            )
        return records

    def create_dataset(self, directory: Path):
        directory.mkdir()
        for shard_number in range(3):
            write_shard(
                directory / f"train-{shard_number:05d}-of-00003.parquet",
                self.make_records(40, shard_number),
            )

    def test_sampling_is_reproducible_and_not_limited_to_shard_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "dataset"
            self.create_dataset(input_dir)
            first_summary, first_sample = collect_validation_sample(
                input_dir,
                sample_size=30,
                seed=2718,
                batch_size=9,
                classifier=FakeClassifier(),
            )
            second_summary, second_sample = collect_validation_sample(
                input_dir,
                sample_size=30,
                seed=2718,
                batch_size=7,
                classifier=FakeClassifier(),
            )

            first_ids = [row["source_row_id"] for row in first_sample]
            second_ids = [row["source_row_id"] for row in second_sample]
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(first_summary["sample_size_actual"], 30)
            self.assertEqual(second_summary["sample_size_actual"], 30)
            self.assertEqual(first_summary["total_rows_scanned"], 120)
            self.assertTrue(
                all(shard["sample_rows"] > 0 for shard in first_summary["shards"])
            )
            for shard in first_summary["shards"]:
                selected_rows = [
                    row["source_row_number_1_based"]
                    for row in first_sample
                    if row["source_shard"] == shard["file"]
                    and "uniform_core" in row["sampling_basis"]
                ]
                self.assertTrue(any(row > 8 for row in selected_rows))

    def test_expected_shard_count_is_checked_before_scanning(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "dataset"
            self.create_dataset(input_dir)
            with self.assertRaisesRegex(ValueError, "Expected 33 Parquet shards"):
                collect_validation_sample(
                    input_dir,
                    sample_size=20,
                    expected_shard_count=33,
                    classifier=FakeClassifier(),
                )

    def test_question_and_response_are_classified_separately(self):
        classifier = FakeClassifier()
        result = classify_record_fields(
            {"question": "Question field", "response": "न्यायालय ने अपील खारिज की।"},
            classifier,
        )
        self.assertEqual(
            classifier.calls,
            ["Question field", "न्यायालय ने अपील खारिज की।"],
        )
        self.assertEqual(result["question"]["language"], "en")
        self.assertEqual(result["response"]["language"], "hi")
        self.assertEqual(result["question"]["classifier_score"], -8.5)
        self.assertFalse(result["question"]["score_is_probability"])

    def test_missing_and_blank_text_are_recorded_without_classification(self):
        classifier = FakeClassifier()
        missing = classify_field(None, classifier)
        blank = classify_field(" \n ", classifier)
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(blank["status"], "blank")
        self.assertEqual(classifier.calls, [])

    def test_context_corruption_flag_is_kept_as_provenance_candidate(self):
        strata = heuristic_strata(
            {
                "context": "Damaged\x05 source context.",
                "question": "What happened?",
                "response": "The appeal was dismissed.",
            }
        )
        self.assertIn("audit_control_or_replacement_flag", strata)

    def test_report_generation_writes_summary_review_template_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "dataset"
            self.create_dataset(input_dir)
            summary, sample = collect_validation_sample(
                input_dir,
                sample_size=24,
                seed=99,
                batch_size=16,
                classifier=FakeClassifier(),
            )
            output_dir = Path(temporary) / "report"
            write_validation_outputs(output_dir, summary, sample)
            expected_files = {
                "language_classification_summary.json",
                "language_review.jsonl",
                "language_review_template.json",
                "language_sampling_report.md",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected_files)
            summary_json = json.loads(
                (output_dir / "language_classification_summary.json").read_text()
            )
            self.assertEqual(summary_json["sample_size_actual"], len(sample))
            self.assertIsNone(summary_json["classifier"]["threshold"])
            review_row = json.loads(
                (output_dir / "language_review.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(
                review_row["context_role"],
                "provenance_only_do_not_use_for_language_label",
            )
            self.assertEqual(review_row["human_question_language"], "")
            report = (output_dir / "language_sampling_report.md").read_text()
            self.assertIn("simple random sample", report)
            self.assertIn("not a calibrated probability", report)

    def test_human_mixed_language_label_is_compared_and_reported(self):
        records = [
            {
                "source_row_id": "row-1",
                "question": "What does कानून mean?",
                "response": "It means law.",
                "question_prediction": {
                    "language": "en",
                    "classifier_score": -1.2,
                },
                "response_prediction": {
                    "language": "en",
                    "classifier_score": -0.4,
                },
                "human_question_language": "mixed",
                "human_response_language": "en",
                "human_mixed_language_status": "question",
                "question_genuinely_english": "partial",
                "response_genuinely_english": "yes",
                "sampling_basis": ["uniform_core"],
            }
        ]
        summary, errors = evaluate_records(records, min_support=1)
        self.assertEqual(summary["fields"]["question"]["accuracy"], 0.0)
        self.assertEqual(
            summary["fields"]["question"]["confusion_matrix"]["counts"]["mixed"][
                "en"
            ],
            1,
        )
        self.assertEqual(summary["human_mixed_language_status_counts"]["question"], 1)
        self.assertEqual(len(errors), 1)

    def test_evaluation_accuracy_confusion_and_per_language_metrics(self):
        records = [
            {
                "source_row_id": f"row-{index}",
                "question": "text",
                "response": "text",
                "question_prediction": {"language": prediction, "classifier_score": 1.0},
                "response_prediction": {"language": "en", "classifier_score": 1.0},
                "human_question_language": gold,
                "human_response_language": "en",
                "question_genuinely_english": "yes",
                "response_genuinely_english": "yes",
                "sampling_basis": ["uniform_core"],
            }
            for index, (gold, prediction) in enumerate(
                [("en", "en"), ("en", "hi"), ("hi", "hi"), ("hi", "hi")]
            )
        ]
        summary, errors = evaluate_records(records, min_support=2)
        question_metrics = summary["fields"]["question"]
        self.assertEqual(question_metrics["accuracy"], 0.75)
        self.assertEqual(question_metrics["n_compared"], 4)
        self.assertEqual(question_metrics["confusion_matrix"]["counts"]["en"]["hi"], 1)
        hi_metrics = next(
            metric for metric in question_metrics["per_language"] if metric["language"] == "hi"
        )
        self.assertAlmostEqual(hi_metrics["precision"], 2 / 3)
        self.assertEqual(hi_metrics["recall"], 1.0)
        self.assertTrue(hi_metrics["reported_as_sufficient_support"])
        self.assertEqual(len(errors), 1)
        self.assertEqual(
            question_metrics["genuinely_english_comparison"]["n_compared"], 4
        )
        self.assertEqual(
            summary["sampling_design_subsets"]["core_stratified_probability_sample"][
                "fields"
            ]["question"]["n_compared"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
