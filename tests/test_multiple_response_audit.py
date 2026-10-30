from __future__ import annotations

import csv
import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_pipeline.audit.multiple_response_audit import (
    _random_rank,
    question_group_id,
    response_variant_id,
    run_multiple_response_audit,
    select_review_groups,
)
from scripts.data_pipeline.audit.run_audit import normalize_for_comparison, stable_hash


def pass_record(
    index: int,
    question: str,
    response: str,
    *,
    language: str = "en",
    quality_flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source_row_id": f"rev:train-00000-of-00001.parquet:{index}",
        "dataset_revision": "rev",
        "source_shard": "train-00000-of-00001.parquet",
        "source_row_number_1_based": index,
        "outcome": "PASS",
        "reason_code": f"SUPPORTED_{language.upper()}_{language.upper()}",
        "secondary_reason_codes": [],
        "question": question,
        "response": response,
        "quality_flags": quality_flags or [],
        "record_fingerprint": f"record-{index}",
        "context_sha256": f"context-{index}",
        "question_language": language,
        "question_language_normalized": language,
        "question_prediction_status": "classified",
        "question_classifier_score": -1.25,
        "response_language": language,
        "response_language_normalized": language,
        "response_prediction_status": "classified",
        "response_classifier_score": -2.5,
    }


class MultipleResponseAuditTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, list[dict[str, object]]]:
        stage3_dir = root / "stage3"
        pass_dir = stage3_dir / "outcome_partitions" / "PASS"
        pass_dir.mkdir(parents=True)
        records = [
            pass_record(1, "What is the rule?", "The rule applies."),
            pass_record(2, "What  is the rule?\n", "The   rule applies.", quality_flags=["sample_flag"]),
            pass_record(3, "What is the rule?", "The rule does not apply."),
            pass_record(4, "Cafe\u0301?", "No.", language="hi"),
            pass_record(5, "Café?", "no", language="hi"),
            pass_record(6, "Single question?", "Single answer."),
        ]
        pq.write_table(pa.Table.from_pylist(records), pass_dir / "train-00000-of-00001.parquet")
        manifest = {
            "train-00000-of-00001.parquet": {
                "rows": 6,
                "bytes": 123,
                "sha256": "same-source-hash",
            }
        }
        (stage3_dir / "source_manifest_before.json").write_text(json.dumps(manifest), encoding="utf-8")
        (stage3_dir / "source_manifest_after.json").write_text(json.dumps(manifest), encoding="utf-8")
        summary = {
            "dataset_revision": "rev",
            "mode": "full_corpus",
            "all_shards_fully_processed": True,
            "raw_source_unchanged": True,
            "source_metadata_rows": 6,
            "outcome_counts": {"PASS": 6, "QUARANTINE": 0, "REJECT": 0},
        }
        (stage3_dir / "stage3_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return stage3_dir, records

    def test_normalization_and_ids_reuse_existing_audit_rules(self):
        self.assertEqual(normalize_for_comparison("What  is\nthe rule?"), "What is the rule?")
        self.assertEqual(
            question_group_id("What  is\nthe rule?"),
            "qgrp_" + stable_hash(["What is the rule?"]),
        )
        self.assertEqual(response_variant_id("Answer  here."), response_variant_id("Answer\nhere."))
        self.assertNotEqual(response_variant_id("Answer here."), response_variant_id("answer here."))
        self.assertNotEqual(response_variant_id("Answer here."), response_variant_id("Answer here"))
        self.assertEqual(question_group_id("Cafe\u0301?"), question_group_id("Café?"))

    def test_full_audit_counts_variants_and_preserves_original_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage3_dir, _ = self.make_fixture(root)
            output_dir = root / "multiple-response-audit"
            summary = run_multiple_response_audit(
                stage3_dir=stage3_dir,
                output_dir=output_dir,
                batch_size=2,
                sample_size=2,
                seed=42,
            )
            self.assertEqual(summary["pass_records_examined"], 6)
            self.assertEqual(summary["distinct_question_groups"], 3)
            self.assertEqual(summary["multiple_response_group_count"], 2)
            self.assertEqual(summary["records_in_multiple_response_groups"], 5)
            self.assertEqual(summary["response_variants_in_multiple_response_groups"], 4)
            self.assertEqual(summary["distinct_response_count_distribution"], {"1": 1, "2": 2})

            artifact = output_dir / "multiple_response_groups.jsonl.gz"
            with gzip.open(artifact, "rt", encoding="utf-8") as handle:
                groups = [json.loads(line) for line in handle]
            self.assertEqual(len(groups), 2)
            rule_group = next(group for group in groups if group["normalized_question"] == "What is the rule?")
            self.assertEqual(rule_group["group_size"], 3)
            self.assertEqual(
                sorted(variant["response_frequency"] for variant in rule_group["response_variants"]),
                [1, 2],
            )
            original_question_texts = {
                record["question_text"]
                for variant in rule_group["response_variants"]
                for record in variant["source_records"]
            }
            self.assertIn("What  is the rule?\n", original_question_texts)
            repeated_variant = next(
                variant for variant in rule_group["response_variants"] if variant["response_frequency"] == 2
            )
            self.assertEqual(repeated_variant["normalized_response"], "The rule applies.")
            self.assertIn("sample_flag", repeated_variant["source_records"][1]["quality_flags"])
            self.assertEqual(repeated_variant["source_records"][0]["source_shard"], "train-00000-of-00001.parquet")

            with (output_dir / "multiple_response_review_sample.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["group_id"] for row in rows}), 2)
            for row in rows:
                self.assertEqual(row["review_disposition"], "")
                self.assertEqual(row["review_notes"], "")
                self.assertEqual(row["reviewer_id"], "")
            self.assertTrue((output_dir / "multiple_response_report.md").is_file())

    def test_sampling_is_reproducible_unique_and_includes_random_remainder(self):
        with sqlite3.connect(":memory:") as connection:
            connection.execute(
                """CREATE TABLE multiple_groups (
                    group_key TEXT PRIMARY KEY, group_size INTEGER,
                    distinct_response_count INTEGER, min_response_characters INTEGER,
                    max_response_characters INTEGER, distinct_surface_forms INTEGER,
                    random_rank TEXT, has_hindi_or_mixed INTEGER
                )"""
            )
            for index in range(15):
                group_key = f"group-{index:02d}"
                variants = 2 + (index % 4)
                min_chars = 20
                max_chars = 160 if index % 3 == 0 else 40
                surface_forms = 3 if index % 2 == 0 else 1
                has_hindi = int(index % 5 == 0)
                connection.execute(
                    "INSERT INTO multiple_groups VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        group_key,
                        variants + 1,
                        variants,
                        min_chars,
                        max_chars,
                        surface_forms,
                        _random_rank(20261030, f"qgrp_{group_key}"),
                        has_hindi,
                    ),
                )
            first, first_meta = select_review_groups(connection, sample_size=10)
            second, second_meta = select_review_groups(connection, sample_size=10)
            self.assertEqual([row["group_id"] for row in first], [row["group_id"] for row in second])
            self.assertEqual(first_meta, second_meta)
            self.assertEqual(len(first), 10)
            self.assertEqual(len({row["group_id"] for row in first}), 10)
            self.assertTrue(any(row["selection_type"] == "random_remaining" for row in first))
            self.assertTrue(any(row["selection_type"] == "hindi_or_mixed_language_prediction" for row in first))
            self.assertFalse(first_meta["targeted_categories_are_probability_samples"])
            self.assertIsNotNone(first_meta["random_remaining_inclusion_probability"])


if __name__ == "__main__":
    unittest.main()
