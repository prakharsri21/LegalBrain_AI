"""Basic input validation and prompt/output handling for LegalBrain AI."""

import re
from dataclasses import dataclass


SYSTEM_PROMPT_VERSION = "legalbrain-question-only-v2"

DISCLAIMER = (
    "LegalBrain AI is an experimental demo. It may make mistakes, and its "
    "information may be outdated. Verify important legal information against "
    "current official sources. This is not legal advice."
)

SYSTEM_PROMPT = """
You are LegalBrain AI, an experimental chatbot specializing in Indian law.

Answer the user's legal question directly, clearly, and concisely using
your existing model knowledge. You do not have access to live legal
databases, websites, or current legal updates.

When a question refers to an "Article" by number without naming another
legal document, assume it refers to the Constitution of India unless
the conversation indicates otherwise.

Do not give a generic answer when the question has a reasonably clear
meaning in Indian law. Explain the relevant legal provision and its
main purpose. If the reference remains genuinely ambiguous, ask a
brief clarifying question rather than inventing an answer.

Never invent statutory provisions, legal citations, cases, or amendments.
State uncertainty when appropriate. Laws may change, so do not claim
to have verified that information is current.

Provide general legal information, not personalized legal advice.
Do not reveal private reasoning; provide only your final answer.
""".strip()


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str = ""


def validate_request(question: str) -> GuardrailDecision:
    """Reject empty or excessively long questions."""
    if not isinstance(question, str) or not question.strip():
        return GuardrailDecision(
            allowed=False,
            reason="Please enter a legal question.",
        )

    if len(question.strip()) > 3000:
        return GuardrailDecision(
            allowed=False,
            reason="Your question is too long. Please keep it under 3,000 characters.",
        )

    return GuardrailDecision(allowed=True)


def build_messages(question: str) -> list[dict[str, str]]:
    """Build a question-only chat prompt. No context or retrieval is used."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{question.strip()} /no_think",
        },
    ]


def extract_visible_answer(raw_output: str) -> str:
    """Remove Qwen reasoning tags and return only the visible answer."""
    if not raw_output:
        return ""

    answer = raw_output.strip()

    # Never display an incomplete reasoning block.
    if "<think>" in answer and "</think>" not in answer:
        return ""

    # Remove complete reasoning blocks, including multiline blocks.
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL,
    )

    # If a reasoning block was started but not completely removed, fail closed.
    if "<think>" in answer or "</think>" in answer:
        return ""

    return answer.strip()