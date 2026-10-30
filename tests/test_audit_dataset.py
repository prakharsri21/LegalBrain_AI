import unittest

from scripts.data_pipeline.audit.run_audit import (
    get_record_flags,
    length_bucket,
    normalize_for_comparison,
    observe_field,
    new_field_stats,
    stable_hash,
)


class AuditDatasetTests(unittest.TestCase):
    def test_whitespace_and_unicode_normalization_for_comparison(self):
        self.assertEqual(
            normalize_for_comparison("  Article\n   32  "),
            "Article 32",
        )
        self.assertEqual(normalize_for_comparison(None), "")

    def test_hash_is_stable_after_whitespace_normalization(self):
        self.assertEqual(
            stable_hash(["Article 32", "Answer here"]),
            stable_hash([" Article\n32 ", "Answer   here"]),
        )
        self.assertNotEqual(
            stable_hash(["Article 32", "Answer A"]),
            stable_hash(["Article 32", "Answer B"]),
        )

    def test_length_bucket_boundaries(self):
        self.assertEqual(length_bucket(0), "0")
        self.assertEqual(length_bucket(20), "1-20")
        self.assertEqual(length_bucket(21), "21-50")
        self.assertEqual(length_bucket(8_000), "4001-8000")
        self.assertEqual(length_bucket(8_001), "8001+")

    def test_field_statistics_count_null_blank_and_length(self):
        stats = new_field_stats()
        observe_field(stats, None)
        observe_field(stats, "   ")
        observe_field(stats, "Indian law")

        self.assertEqual(stats["null_values"], 1)
        self.assertEqual(stats["blank_values"], 1)
        self.assertEqual(stats["string_values"], 2)
        self.assertEqual(stats["max_characters"], len("Indian law"))

    def test_context_dependent_question_is_flagged(self):
        record = {
            "context": "A sample judgment text.",
            "question": "What did the court decide in this case?",
            "response": "The court dismissed the appeal.",
        }
        self.assertIn(
            "possible_context_dependent_question",
            get_record_flags(record),
        )

    def test_clear_question_answer_record_has_no_flags(self):
        record = {
            "context": "Article 32 protects constitutional remedies.",
            "question": "What does Article 32 of the Constitution provide?",
            "response": (
                "It provides the right to constitutional remedies for "
                "enforcement of Fundamental Rights."
            ),
        }
        self.assertEqual(get_record_flags(record), [])


    def test_form_feed_is_counted_separately_not_flagged_as_control(self):
        from scripts.data_pipeline.audit.run_audit import (
            get_record_flags,
            observe_field,
            new_field_stats,
        )

        context = "First page text\fSecond page text\fThird page text"
        stats = new_field_stats()
        observe_field(stats, context)

        self.assertEqual(stats["values_with_control_characters"], 0)
        self.assertEqual(stats["values_with_form_feed_page_break"], 1)
        self.assertEqual(stats["form_feed_page_break_occurrences"], 2)

        record = {
            "context": context,
            "question": "What did the court decide?",
            "response": "The court dismissed the appeal.",
        }
        self.assertNotIn("control_characters_context", get_record_flags(record))

    def test_placeholder_response_is_flagged(self):
        record = {
            "context": "A judgment passage.",
            "question": "What does the court decide?",
            "response": "The answer is not available.",
        }
        self.assertIn("placeholder_response", get_record_flags(record))

    def test_short_substantive_response_is_not_placeholder_flagged(self):
        record = {
            "context": "A judgment passage.",
            "question": "Who prevailed?",
            "response": "The appellant.",
        }
        self.assertNotIn("placeholder_response", get_record_flags(record))

    def test_statement_like_question_is_flagged(self):
        record = {
            "context": "A judgment passage.",
            "question": (
                "The trial judge correctly determined the plaintiffs owned the property."
            ),
            "response": "Yes, the trial judge correctly determined ownership.",
        }
        self.assertIn("statement_like_question", get_record_flags(record))

    def test_instruction_style_prompt_is_not_statement_flagged(self):
        record = {
            "context": "A judgment passage.",
            "question": "Explain the rights of a cultivating tenant under Section 53.",
            "response": "A cultivating tenant may have specified rights under the Act.",
        }
        self.assertNotIn("statement_like_question", get_record_flags(record))

    def test_non_question_fragment_is_not_automatically_a_statement(self):
        record = {
            "context": "A judgment passage.",
            "question": "Rights of a cultivating tenant under Section 53",
            "response": "A cultivating tenant may have specified rights.",
        }
        self.assertNotIn("statement_like_question", get_record_flags(record))

    def test_missing_legal_reference_number_is_flagged(self):
        record = {
            "context": "An excerpt from the Tripura Buildings Lease and Rent Control Rules.",
            "question": "Explain the procedure under Section of the Tripura Rules.",
            "response": "The appeal must be filed with the relevant authority.",
        }
        self.assertIn("missing_legal_reference_number", get_record_flags(record))

    def test_section_with_number_is_not_flagged_as_missing_reference(self):
        record = {
            "context": "Section 53 describes tenant rights.",
            "question": "Explain the rights under Section 53 of the Act.",
            "response": "Section 53 provides a right subject to the Act's requirements.",
        }
        self.assertNotIn("missing_legal_reference_number", get_record_flags(record))

    def test_generic_section_under_consideration_is_not_flagged(self):
        record = {
            "context": "A judgment passage.",
            "question": "What does the section under consideration require?",
            "response": "It requires notice before termination.",
        }
        self.assertNotIn("missing_legal_reference_number", get_record_flags(record))

    def test_ordinary_legal_phrases_are_not_missing_reference_flags(self):
        questions = (
            "How is the rule of law upheld?",
            "What protections are guaranteed under the rule of law?",
            "What section of the Act allowed the acquisition?",
            "Under which section of the RTI Act may information be withheld?",
            "Does the acquisition section of the Act include easementary rights?",
            "Does a specific clause in the settlement deed affect the assessment?",
        )
        for question in questions:
            with self.subTest(question=question):
                record = {
                    "context": "A judgment passage.",
                    "question": question,
                    "response": "The relevant provision applies.",
                }
                self.assertNotIn(
                    "missing_legal_reference_number", get_record_flags(record)
                )

    def test_explicit_blank_legal_reference_is_flagged(self):
        record = {
            "context": "A judgment passage.",
            "question": "Explain the effect of Section ___ of the Code.",
            "response": "The provision governs the procedure.",
        }
        self.assertIn("missing_legal_reference_number", get_record_flags(record))

    def test_missing_reference_in_context_is_flagged(self):
        record = {
            "context": "In terms of Section of the Code, the court may grant relief.",
            "question": "What relief may the court grant?",
            "response": "The court may grant the relief permitted by law.",
        }
        self.assertIn("missing_legal_reference_number", get_record_flags(record))

    def test_rule_of_law_in_context_is_not_a_missing_reference(self):
        record = {
            "context": "Under the rule of law, public authorities must follow due process.",
            "question": "What must public authorities follow?",
            "response": "They must follow due process.",
        }
        self.assertNotIn("missing_legal_reference_number", get_record_flags(record))

    def test_null_and_malformed_fields_are_flagged(self):
        record = {
            "context": None,
            "question": " ",
            "response": None,
        }
        flags = get_record_flags(record)
        self.assertIn("missing_context", flags)
        self.assertIn("blank_question", flags)
        self.assertIn("missing_response", flags)


if __name__ == "__main__":
    unittest.main()
