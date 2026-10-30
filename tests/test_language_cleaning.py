from __future__ import annotations

import hashlib
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_pipeline.clean.language_policy import (
    LanguagePolicy,
    build_language_outcome_record,
    decide_language_outcome,
    normalize_language_code,
)
from scripts.data_pipeline.clean.run_language_clean import (
    _write_stage3_report,
    reservoir_sample_shard,
    run_language_cleaning,
)


def approved_policy() -> LanguagePolicy:
    # Deliberately test-only raw-score floors; production values require validation.
    return LanguagePolicy(
        language_gate_approved=True,
        decision_mode="score_floor",
        validation_reference="test-fixture-only",
        approved_by="unit-test",
        minimum_classifier_score_by_field={"question": 5.0, "response": 5.0},
    )


def approved_label_policy() -> LanguagePolicy:
    return LanguagePolicy(
        language_gate_approved=True,
        decision_mode="label_based",
        validation_reference="reviewed-pilot-report",
        approved_by="user-approved-label-based-policy",
        minimum_classifier_score_by_field={"question": None, "response": None},
    )


def prediction(language: str, score: float = 10.0, status: str = "classified") -> dict[str, Any]:
    return {
        "status": status,
        "language": language,
        "classifier_score": score,
        "score_is_probability": False,
    }


class FakeClassifier:
    def __init__(self, result: tuple[str, float] = ("en", 10.0)) -> None:
        self.result = result
        self.seen: list[str] = []

    def classify(self, text: str) -> tuple[str, float]:
        self.seen.append(text)
        return self.result


class RuleClassifier:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def classify(self, text: str) -> tuple[str, float]:
        self.seen.append(text)
        if "हिंदी" in text:
            return "hi", -1000.0
        if text.startswith("Unsupported"):
            return "zh", 100.0
        return "en", -500.0


class LanguagePolicyTests(unittest.TestCase):
    def assert_decision(self, expected_outcome: str, expected_reason: str, **kwargs: Any) -> dict[str, Any]:
        decision = decide_language_outcome(policy=approved_policy(), **kwargs)
        self.assertEqual(decision["outcome"], expected_outcome)
        self.assertEqual(decision["reason_code"], expected_reason)
        self.assertIn(decision["outcome"], {"PASS", "QUARANTINE", "REJECT"})
        return decision

    def test_english_question_and_response_pass(self):
        self.assert_decision(
            "PASS",
            "SUPPORTED_EN_EN",
            question_prediction=prediction("en"),
            response_prediction=prediction("eng"),
        )

    def test_hindi_question_and_response_pass_with_iso_aliases(self):
        self.assert_decision(
            "PASS",
            "SUPPORTED_HI_HI",
            question_prediction=prediction("hin"),
            response_prediction=prediction("hi"),
        )

    def test_mixed_field_is_quarantined_even_with_component_evidence(self):
        decision = self.assert_decision(
            "QUARANTINE",
            "UNVERIFIED_MIXED_LANGUAGE",
            question_prediction=prediction("mixed"),
            response_prediction=prediction("en"),
            verified_mixed_components={"question": ["eng", "hin"]},
            mixed_review_reference="review-batch-17",
        )
        self.assertTrue(decision["language_gate_operational"])

    def test_mixed_label_without_known_constituents_is_quarantined(self):
        self.assert_decision(
            "QUARANTINE",
            "UNVERIFIED_MIXED_LANGUAGE",
            question_prediction=prediction("mixed"),
            response_prediction=prediction("en"),
        )

    def test_english_question_hindi_response_is_quarantined(self):
        self.assert_decision(
            "QUARANTINE",
            "CROSS_LANGUAGE_PAIR_REVIEW",
            question_prediction=prediction("en"),
            response_prediction=prediction("hi"),
        )

    def test_hindi_question_english_response_is_quarantined(self):
        self.assert_decision(
            "QUARANTINE",
            "CROSS_LANGUAGE_PAIR_REVIEW",
            question_prediction=prediction("hi"),
            response_prediction=prediction("en"),
        )

    def test_unsupported_question_language_is_quarantined(self):
        self.assert_decision(
            "QUARANTINE",
            "UNSUPPORTED_LANGUAGE_REVIEW",
            question_prediction=prediction("zh"),
            response_prediction=prediction("en"),
        )

    def test_unsupported_response_language_is_quarantined(self):
        self.assert_decision(
            "QUARANTINE",
            "UNSUPPORTED_LANGUAGE_REVIEW",
            question_prediction=prediction("en"),
            response_prediction=prediction("ur"),
        )

    def test_low_raw_classifier_score_is_quarantined(self):
        decision = self.assert_decision(
            "QUARANTINE",
            "LOW_CLASSIFIER_SCORE",
            question_prediction=prediction("en", 4.99),
            response_prediction=prediction("en"),
        )
        self.assertIn("LOW_CLASSIFIER_SCORE", decision["secondary_reason_codes"])

    def test_probability_like_score_metadata_is_not_accepted_as_langid_score(self):
        probability_prediction = prediction("en")
        probability_prediction["score_is_probability"] = True
        self.assert_decision(
            "QUARANTINE",
            "LANGUAGE_METADATA_MISSING_OR_INVALID",
            question_prediction=probability_prediction,
            response_prediction=prediction("en"),
        )

    def test_missing_language_metadata_is_quarantined(self):
        self.assert_decision(
            "QUARANTINE",
            "LANGUAGE_METADATA_MISSING_OR_INVALID",
            question_prediction=None,
            response_prediction=prediction("en"),
        )

    def test_unknown_code_is_not_treated_as_supported(self):
        self.assertIsNone(normalize_language_code("zz"))
        self.assert_decision(
            "QUARANTINE",
            "UNKNOWN_OR_INVALID_LANGUAGE_CODE",
            question_prediction=prediction("zz"),
            response_prediction=prediction("en"),
        )

    def test_supported_code_aliases_normalize_consistently(self):
        self.assertEqual(normalize_language_code("EN"), "en")
        self.assertEqual(normalize_language_code("eng"), "en")
        self.assertEqual(normalize_language_code("hi"), "hi")
        self.assertEqual(normalize_language_code("hin"), "hi")

    def test_clear_unsupported_language_precedes_other_field_uncertainty(self):
        decision = self.assert_decision(
            "QUARANTINE",
            "UNSUPPORTED_LANGUAGE_REVIEW",
            question_prediction=prediction("zh"),
            response_prediction=prediction("en", 1.0),
        )
        self.assertEqual(len([decision["outcome"]]), 1)
        self.assertIn("LOW_CLASSIFIER_SCORE", decision["secondary_reason_codes"])

    def test_unapproved_gate_quarantines_without_using_scores(self):
        decision = decide_language_outcome(
            prediction("en", 1000), prediction("en", 1000), LanguagePolicy()
        )
        self.assertEqual(decision["outcome"], "QUARANTINE")
        self.assertEqual(decision["reason_code"], "LANGUAGE_GATE_NOT_APPROVED")

    def test_unapproved_gate_keeps_field_metadata_issues_as_secondary_evidence(self):
        decision = decide_language_outcome(None, prediction("en"), LanguagePolicy())
        self.assertEqual(decision["outcome"], "QUARANTINE")
        self.assertIn(
            "LANGUAGE_METADATA_MISSING_OR_INVALID", decision["secondary_reason_codes"]
        )

    def test_approved_label_policy_needs_reference_but_no_score_floors(self):
        with self.assertRaises(ValueError):
            LanguagePolicy(language_gate_approved=True)
        label_policy = LanguagePolicy(
            language_gate_approved=True,
            decision_mode="label_based",
            validation_reference="reviewed-pilot",
            approved_by="user-approved-task",
            minimum_classifier_score_by_field={"question": None, "response": None},
        )
        self.assertTrue(label_policy.is_operational)
        with self.assertRaises(ValueError):
            LanguagePolicy(
                language_gate_approved=True,
                decision_mode="label_based",
                validation_reference="reviewed-pilot",
                approved_by="user-approved-task",
                minimum_classifier_score_by_field={"question": 0.1, "response": None},
            )
        with self.assertRaises(ValueError):
            LanguagePolicy(
                language_gate_approved=True,
                decision_mode="score_floor",
                validation_reference="reviewed-pilot",
                approved_by="user-approved-task",
                minimum_classifier_score_by_field={"question": None, "response": None},
            )

    def test_label_based_policy_does_not_use_raw_score_for_language_outcome(self):
        policy = LanguagePolicy(
            language_gate_approved=True,
            decision_mode="label_based",
            validation_reference="reviewed-pilot",
            approved_by="user-approved-task",
            minimum_classifier_score_by_field={"question": None, "response": None},
        )
        low = decide_language_outcome(
            prediction("en", -5000), prediction("en", -9000), policy
        )
        high = decide_language_outcome(
            prediction("en", 10000), prediction("en", 10000), policy
        )
        self.assertEqual(low["outcome"], "PASS")
        self.assertEqual(low["reason_code"], high["reason_code"])
        self.assertEqual(low["outcome"], high["outcome"])

    def test_outcome_record_retains_provenance_reason_and_separate_quality_signal(self):
        decision = decide_language_outcome(
            prediction("en"), prediction("en"), approved_policy()
        )
        record = build_language_outcome_record(
            dataset_revision="rev",
            source_shard="shard-1.parquet",
            source_row_number_1_based=8,
            question="Question?",
            response="Answer.",
            question_prediction=prediction("en"),
            response_prediction=prediction("en"),
            decision=decision,
            classifier_version="test",
            quality_signals={"statement_like_question": True},
        )
        self.assertEqual(record["source_row_id"], "rev:shard-1.parquet:8")
        self.assertEqual(record["outcome"], "PASS")
        self.assertEqual(record["outcome_scope"], "language_eligibility_only")
        self.assertTrue(record["reason_code"])
        self.assertEqual(record["quality_signals"], {"statement_like_question": True})
        self.assertEqual(record["question_prediction"]["language"], "en")
        self.assertEqual(record["question_prediction"]["score_is_probability"], False)

    def test_quality_signals_do_not_change_language_eligibility(self):
        baseline = decide_language_outcome(
            prediction("en"), prediction("en"), approved_policy()
        )
        record = build_language_outcome_record(
            dataset_revision="rev",
            source_shard="s.parquet",
            source_row_number_1_based=1,
            question="The court did X.",
            response="Yes.",
            question_prediction=prediction("en"),
            response_prediction=prediction("en"),
            decision=baseline,
            classifier_version="test",
            quality_signals={"statement_like_question": True},
        )
        self.assertEqual(record["outcome"], baseline["outcome"])
        self.assertIn("statement_like_question", record["quality_signals"])


class LanguageCleanRunnerTests(unittest.TestCase):
    @staticmethod
    def make_shard(path: Path, shard_name: str, count: int, start: int = 0) -> Path:
        shard_path = path / shard_name
        table = pa.table(
            {
                "context": ["CONTEXT MUST NOT BE USED " + ("हिंदी " * 10)] * count,
                "question": [f"Question {i}?" for i in range(start, start + count)],
                "response": [f"Response {i}." for i in range(start, start + count)],
            }
        )
        pq.write_table(table, shard_path)
        return shard_path

    @staticmethod
    def make_records_shard(path: Path, records: list[dict[str, Any]]) -> Path:
        shard_path = path / "train-00000-of-00001.parquet"
        table = pa.Table.from_pylist(records)
        pq.write_table(table, shard_path)
        return shard_path

    def test_reservoir_sample_is_reproducible_stratified_by_shard_and_not_first_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_a = self.make_shard(root, "train-00000-of-00002.parquet", 100)
            shard_b = self.make_shard(root, "train-00001-of-00002.parquet", 80, 100)
            sample_a, count_a = reservoir_sample_shard(shard_a, 9, batch_size=13, seed=2026)
            sample_b, count_b = reservoir_sample_shard(shard_b, 9, batch_size=17, seed=2026)
            repeat_a, _ = reservoir_sample_shard(shard_a, 9, batch_size=7, seed=2026)
            self.assertEqual([x["source_row_number_1_based"] for x in sample_a], [x["source_row_number_1_based"] for x in repeat_a])
            self.assertEqual((len(sample_a), len(sample_b), count_a, count_b), (9, 9, 100, 80))
            self.assertNotEqual([x["source_row_number_1_based"] for x in sample_a], list(range(1, 10)))
            self.assertEqual(len({x["source_row_number_1_based"] for x in sample_a}), 9)

    def test_sample_run_reconciles_stage3_outputs_and_preserves_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "raw"
            source_dir.mkdir()
            shard = self.make_shard(source_dir, "train-00000-of-00001.parquet", 25)
            before_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
            classifier = FakeClassifier(("en", 8.0))
            output = root / "out"
            summary = run_language_cleaning(
                input_dir=source_dir,
                output_dir=output,
                dataset_revision="fixture-rev",
                policy=LanguagePolicy(),
                batch_size=6,
                sample_per_shard=10,
                seed=11,
                classifier=classifier,
            )
            after_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
            with gzip.open(output / "outcome_ledger.jsonl.gz", "rt", encoding="utf-8") as ledger:
                rows = [json.loads(line) for line in ledger]
            self.assertEqual(before_hash, after_hash)
            self.assertEqual(len(rows), 10)
            self.assertEqual(len({row["source_row_id"] for row in rows}), 10)
            self.assertEqual(summary["processed_record_count"], 10)
            self.assertEqual(summary["outcome_counts"], {"PASS": 0, "QUARANTINE": 10, "REJECT": 0})
            self.assertEqual(sum(summary["outcome_counts"].values()), summary["processed_record_count"])
            self.assertTrue(all("context" not in row for row in rows))
            self.assertTrue(all(row["context_use"].startswith("read_for_provenance") for row in rows))
            self.assertTrue(all(row["context_sha256"] for row in rows))
            self.assertEqual(len(classifier.seen), 20)
            self.assertTrue(all(row["reason_code"] == "LANGUAGE_GATE_NOT_APPROVED" for row in rows))
            self.assertTrue(summary["outcome_ledger_reconciles"])
            self.assertTrue(summary["question_response_field_statistics_reconcile"])
            self.assertEqual(summary["raw_question_response_text_hash_validation"]["question_hash_mismatches"], 0)

    def test_full_scan_refuses_unapproved_policy_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "raw"
            source_dir.mkdir()
            self.make_shard(source_dir, "train-00000-of-00001.parquet", 2)
            output = root / "full"
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                run_language_cleaning(
                    input_dir=source_dir,
                    output_dir=output,
                    policy=LanguagePolicy(),
                    sample_per_shard=None,
                )
            self.assertFalse(output.exists())

    def test_sample_outputs_are_reproducible_and_keep_shard_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "raw"
            source_dir.mkdir()
            self.make_shard(source_dir, "train-00000-of-00002.parquet", 30)
            self.make_shard(source_dir, "train-00001-of-00002.parquet", 20, 30)
            ids_by_run = []
            for suffix in ("one", "two"):
                summary = run_language_cleaning(
                    input_dir=source_dir,
                    output_dir=root / suffix,
                    dataset_revision="rev",
                    policy=LanguagePolicy(),
                    batch_size=5,
                    sample_per_shard=7,
                    seed=123,
                    classifier=FakeClassifier(),
                )
                with gzip.open(root / suffix / "outcome_ledger.jsonl.gz", "rt", encoding="utf-8") as ledger:
                    records = [json.loads(line) for line in ledger]
                ids_by_run.append([record["source_row_id"] for record in records])
                self.assertEqual(summary["processed_record_count"], 14)
                self.assertEqual(summary["source_rows_streamed_for_selection"], 50)
                self.assertEqual(len({record["source_shard"] for record in records}), 2)
                self.assertEqual(
                    {record["sampling_metadata"]["selection_type"] for record in records},
                    {"uniform_per_shard_reservoir"},
                )
            self.assertEqual(ids_by_run[0], ids_by_run[1])

    def test_stage3_rules_duplicate_provenance_and_label_based_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "raw"
            source_dir.mkdir()
            records = [
                {"context": "ctx-a", "question": "Question A?", "response": "Answer A is fully stated."},
                {"context": " ctx-a ", "question": " Question   A? ", "response": "Answer A is fully stated."},
                {"context": "ctx-statement", "question": "The court is correct.", "response": "Yes."},
                {"context": "ctx-placeholder", "question": "What is the answer?", "response": "The answer is not available."},
                {"context": "ctx-page", "question": "What happened on the next page?\f", "response": "The court dismissed the appeal."},
                {"context": "ctx-ref", "question": "Under Section __ of the Act, what applies?", "response": "The rule applies."},
                {"context": "ctx-control", "question": "What happened?", "response": "A bad\x01answer."},
                {"context": "ctx-missing", "question": None, "response": "A response is present."},
                {"context": "ctx-short", "question": "What was the amount?", "response": "Rs 500"},
                {"context": "ctx-context", "question": "According to the passage, what did the court decide?", "response": "It dismissed the appeal."},
                {"context": "ctx-cross", "question": "हिंदी प्रश्न क्या है?", "response": "The response is in English."},
                {"context": "ctx-unsupported", "question": "Unsupported question?", "response": "An English response."},
                {"context": "ctx-replacement", "question": "What does the text say?", "response": "The damaged word is \ufffd."},
                {"context": "Under Section __ of the Act", "question": "What legal rule governs the claim?", "response": "The rule governs the claim."},
                {"context": "ctx-short-pass", "question": "What happened?", "response": "Yes."},
                {"context": "A source\x01 artifact", "question": "What is the rule?", "response": "The rule is described here."},
            ]
            self.make_records_shard(source_dir, records)
            output = root / "stage3"
            classifier = RuleClassifier()
            summary = run_language_cleaning(
                input_dir=source_dir,
                output_dir=output,
                dataset_revision="fixture-rev",
                policy=approved_label_policy(),
                batch_size=4,
                classifier=classifier,
            )

            self.assertEqual(
                summary["outcome_counts"],
                {"PASS": 6, "QUARANTINE": 8, "REJECT": 2},
                summary["primary_reason_code_counts"],
            )
            self.assertEqual(summary["duplicate_occurrence_count"], 1)
            self.assertEqual(summary["duplicate_group_count"], 1)
            self.assertTrue(summary["question_response_field_statistics_reconcile"])
            self.assertEqual(summary["text_integrity_occurrences"]["question"]["form_feed"], 1)
            self.assertEqual(summary["text_integrity_occurrences"]["response"]["control_characters"], 1)
            self.assertEqual(summary["missing_legal_reference_match_field_counts"]["context"], 1)
            self.assertEqual(summary["missing_legal_reference_match_field_counts"]["question"], 1)
            self.assertTrue(summary["outcome_ledger_reconciles"])
            self.assertTrue(summary["raw_question_response_text_hash_validation"]["records_checked"] == len(records))
            self.assertTrue(summary["memory_model"]["fingerprint_index_disk_backed"])

            reject_files = list((output / "outcome_partitions" / "REJECT").glob("*.parquet"))
            rejected = pq.read_table(reject_files).to_pylist()
            duplicate = next(row for row in rejected if row["reason_code"] == "DUPLICATE_EXACT_FULL_RECORD")
            self.assertEqual(
                duplicate["duplicate_of_source_row_id"],
                "fixture-rev:train-00000-of-00001.parquet:1",
            )

            pass_files = list((output / "outcome_partitions" / "PASS").glob("*.parquet"))
            passed = pq.read_table(pass_files).to_pylist()
            short_row = next(row for row in passed if row["response"] == "Rs 500")
            self.assertIn("short_response", short_row["quality_flags"])
            self.assertEqual(summary["per_field_prediction_categories"]["question"]["unsupported"], 1)
            self.assertEqual(summary["per_field_prediction_categories"]["question"]["hindi"], 1)

            summary["test_results"] = {
                "tests_run": 84,
                "passed": 84,
                "failed": 0,
                "command": "uv run python -m unittest discover -s tests -q",
                "syntax_check_passed": True,
                "run_timing": "before data scan",
            }
            report_path = output / "stage3_report.md"
            _write_stage3_report(summary, report_path)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("84 passed; 0 failed", report)
            self.assertIn("before data scan", report)

    def test_label_based_gate_does_not_reject_unsupported_prediction(self):
        decision = decide_language_outcome(
            prediction("zh", 1000), prediction("en", 1000), approved_label_policy()
        )
        self.assertEqual(decision["outcome"], "QUARANTINE")
        self.assertEqual(decision["reason_code"], "UNSUPPORTED_LANGUAGE_REVIEW")


if __name__ == "__main__":
    unittest.main()
