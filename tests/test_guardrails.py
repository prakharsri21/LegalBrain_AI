import unittest

from app.services.guardrails import (
    DISCLAIMER,
    SYSTEM_PROMPT_VERSION,
    build_messages,
    extract_visible_answer,
    validate_request,
)


class GuardrailTests(unittest.TestCase):
    def test_empty_question_is_rejected(self):
        self.assertFalse(validate_request("").allowed)
        self.assertFalse(validate_request("   ").allowed)

    def test_question_over_limit_is_rejected(self):
        self.assertFalse(validate_request("x" * 3001).allowed)

    def test_valid_question_is_accepted(self):
        self.assertTrue(
            validate_request("What does Article 32 provide?").allowed
        )

    def test_prompt_contains_question_without_context(self):
        messages = build_messages("What does Article 32 provide?")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Article 32", messages[1]["content"])
        self.assertNotIn("Context:", messages[1]["content"])
        self.assertIn("/no_think", messages[1]["content"])

    def test_reasoning_block_is_removed(self):
        raw = "<think>private reasoning</think>\nArticle 32 provides a remedy."
        self.assertEqual(
            extract_visible_answer(raw),
            "Article 32 provides a remedy.",
        )

    def test_incomplete_reasoning_block_returns_empty(self):
        self.assertEqual(
            extract_visible_answer("<think>unfinished reasoning"),
            "",
        )

    def test_empty_output_returns_empty(self):
        self.assertEqual(extract_visible_answer(""), "")

    def test_disclaimer_and_prompt_version_exist(self):
        self.assertIn("outdated", DISCLAIMER)
        self.assertTrue(SYSTEM_PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()