import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.data_pipeline.audit.complete_language_review_pilot import (
    merge_completed_csv,
)
from scripts.data_pipeline.audit.create_language_review_pilot import (
    DEFAULT_TEMPLATE,
    LABEL_FIELDS,
    create_pilot_records,
    read_jsonl,
    write_pilot_outputs,
)
from scripts.data_pipeline.audit.evaluate_language_review import read_jsonl as read_eval_jsonl


class LanguageReviewPilotTests(unittest.TestCase):
    def make_records(self):
        records = []
        languages = ["hi", "zh", "fr", "la", "mr", "ne", "jv", "ur"]
        error_strata = [
            "audit_control_or_replacement_flag",
            "legal_citation",
            "name_like_capitalized_sequence",
            "quotation_mark",
            "long_text_at_least_1000_chars",
            "replacement_character",
            "suspicious_control_character",
        ]
        for source_type, count in (("uniform_core", 800), ("supplemental_targeted", 200)):
            for index in range(count):
                global_index = index if source_type == "uniform_core" else index + 800
                row_number = global_index // 33 + 1
                shard_number = global_index % 33
                strata = [
                    "predicted_question_language:en",
                    "predicted_response_language:en",
                ]
                if source_type == "supplemental_targeted":
                    language = languages[index % len(languages)]
                    strata[0] = f"predicted_question_language:{language}"
                    strata[1] = f"predicted_response_language:{language}"
                    if index % 2 == 0:
                        strata.append("mixed_language_looking")
                    if index % 3 == 0:
                        strata.append("question_response_prediction_disagreement")
                    if index % 4 == 0:
                        strata.append("short_text_under_40_chars")
                    if index % 5 == 0:
                        strata.append(error_strata[(index // 5) % len(error_strata)])
                records.append(
                    {
                        "source_row_id": f"revision:train-{shard_number:05d}:{row_number}",
                        "source_shard": f"train-{shard_number:05d}-of-00033.parquet",
                        "source_row_number_1_based": row_number,
                        "dataset_revision": "revision",
                        "question": f"What legal rule applies to record {index}?",
                        "response": f"The response for record {index} is in English.",
                        "context": f"Do not use this context {index}",
                        "context_role": "provenance_only_do_not_use_for_language_label",
                        "context_sha256": f"context-hash-{index}",
                        "field_lengths_chars": {
                            "context": 30,
                            "question": 40,
                            "response": 41,
                        },
                        "candidate_strata": sorted(set(strata)),
                        "sampling_basis": [source_type],
                        "uniform_core_inclusion_probability": 0.0013
                        if source_type == "uniform_core"
                        else None,
                        "supplemental_selection_strata": []
                        if source_type == "uniform_core"
                        else ["original targeted stratum"],
                        "audit_flags": [],
                        "question_prediction": {
                            "status": "classified",
                            "language": strata[0].split(":", 1)[1],
                            "classifier_score": -123.45 - index,
                            "score_is_probability": False,
                        },
                        "response_prediction": {
                            "status": "classified",
                            "language": strata[1].split(":", 1)[1],
                            "classifier_score": -678.9 - index,
                            "score_is_probability": False,
                        },
                        "human_question_language": "",
                        "human_response_language": "",
                        "human_mixed_language_status": "",
                        "question_genuinely_english": "",
                        "response_genuinely_english": "",
                        "reviewer_id": "",
                        "review_notes": "",
                    }
                )
        return records

    def test_pilot_selection_is_reproducible_stratified_and_unique(self):
        records = self.make_records()
        prediction_counts = {
            "question": {"en": 8_000, "hi": 300, "zh": 140, "fr": 8, "la": 51},
            "response": {"en": 8_900, "hi": 300, "zh": 490, "fr": 8, "la": 110},
        }
        first, first_summary = create_pilot_records(
            records, seed=556677, prediction_counts=prediction_counts
        )
        second, second_summary = create_pilot_records(
            records, seed=556677, prediction_counts=prediction_counts
        )
        first_ids = [record["source_row_id"] for record in first]
        second_ids = [record["source_row_id"] for record in second]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first, second)
        self.assertEqual(first_summary["pilot_record_count"], 500)
        self.assertEqual(first_summary["pilot_probability_sample_count"], 400)
        self.assertEqual(first_summary["pilot_targeted_supplemental_count"], 100)
        self.assertEqual(len(set(first_ids)), 500)
        self.assertEqual(
            Counter(
                record["pilot_selection"]["pilot_selection_type"] for record in first
            ),
            {"probability_sample": 400, "targeted_supplemental": 100},
        )
        targeted_groups = first_summary["pilot_targeted_group_counts"]
        self.assertEqual(targeted_groups["uncommon_predicted_language"], 40)
        self.assertEqual(targeted_groups["mixed_language_indicator"], 25)
        self.assertEqual(targeted_groups["short_field"], 15)
        self.assertEqual(targeted_groups["potential_classifier_error"], 20)

    def test_source_metadata_is_preserved_and_context_is_omitted(self):
        records = self.make_records()
        pilot, _ = create_pilot_records(records, seed=101)
        original_by_id = {record["source_row_id"]: record for record in records}
        for selected in pilot:
            original = original_by_id[selected["source_row_id"]]
            self.assertNotIn("context", selected)
            for key, value in original.items():
                if key != "context":
                    self.assertEqual(selected[key], value, key)
            self.assertIn(selected["sampling_basis"][0], {"uniform_core", "supplemental_targeted"})
            self.assertEqual(
                selected["question_prediction"]["classifier_score"],
                original["question_prediction"]["classifier_score"],
            )
            self.assertEqual(
                selected["context_role"],
                "provenance_only_do_not_use_for_language_label",
            )

    def test_csv_jsonl_and_template_schema_are_valid(self):
        records = self.make_records()
        pilot, summary = create_pilot_records(records, seed=73)
        template = json.loads(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pilot"
            write_pilot_outputs(output_dir, pilot, summary, template, seed=73)
            jsonl_records = read_jsonl(output_dir / "language_review_pilot.jsonl")
            with (output_dir / "language_review_pilot.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                csv_reader = csv.DictReader(handle)
                csv_rows = list(csv_reader)
                csv_fields = set(csv_reader.fieldnames or [])
            self.assertEqual(len(jsonl_records), 500)
            self.assertEqual(len(csv_rows), 500)
            self.assertNotIn("context", jsonl_records[0])
            self.assertNotIn("context", csv_fields)
            self.assertTrue(set(LABEL_FIELDS).issubset(csv_fields))
            self.assertIn("question_prediction_classifier_score", csv_fields)
            self.assertIn("sampling_basis", csv_fields)
            self.assertIn("context_sha256", csv_fields)
            self.assertEqual(
                json.loads((output_dir / "language_review_template.json").read_text()),
                template,
            )
            instructions = (
                output_dir / "language_review_pilot_instructions.md"
            ).read_text()
            for field in LABEL_FIELDS:
                self.assertIn(f"`{field}`", instructions)
            self.assertIn("Do not use the source `context`", instructions)

    def test_completed_csv_merges_only_labels_and_preserves_evaluator_schema(self):
        records = self.make_records()
        pilot, summary = create_pilot_records(records, seed=88)
        template = json.loads(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pilot"
            write_pilot_outputs(output_dir, pilot, summary, template, seed=88)
            csv_path = output_dir / "language_review_pilot_completed.csv"
            with (output_dir / "language_review_pilot.csv").open(
                encoding="utf-8-sig", newline=""
            ) as source, csv_path.open("w", encoding="utf-8-sig", newline="") as target:
                reader = csv.DictReader(source)
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    row.update(
                        {
                            "human_question_language": "en",
                            "human_response_language": "mixed",
                            "human_mixed_language_status": "response",
                            "question_genuinely_english": "yes",
                            "response_genuinely_english": "partial",
                            "reviewer_id": "reviewer-1",
                            "review_notes": "pilot test",
                        }
                    )
                    writer.writerow(row)

            completed_jsonl = output_dir / "language_review_pilot_completed.jsonl"
            count = merge_completed_csv(
                output_dir / "language_review_pilot.jsonl",
                csv_path,
                completed_jsonl,
                output_dir / "language_review_template.json",
            )
            completed = read_eval_jsonl(completed_jsonl)
            self.assertEqual(count, 500)
            self.assertEqual(len(completed), 500)
            self.assertEqual(completed[0]["human_question_language"], "en")
            self.assertEqual(completed[0]["human_mixed_language_status"], "response")
            self.assertIn("sampling_basis", completed[0])
            self.assertIn("question_prediction", completed[0])
            self.assertNotIn("context", completed[0])


if __name__ == "__main__":
    unittest.main()
