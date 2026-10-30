"""Conservative English/Hindi eligibility decisions for Stage 3.

The approved V1 mode uses predicted labels only: English-English and
Hindi-Hindi pairs pass the language check; all other language combinations are
quarantined. Raw langid scores remain uncalibrated evidence and do not affect
this mode's decisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SUPPORTED_LANGUAGES = frozenset({"en", "hi"})
LANGUAGE_ALIASES = {"eng": "en", "hin": "hi"}

# Codes emitted by the langid model bundled by the current project dependency.
# Unknown codes are quarantined rather than assumed to be a known unsupported
# language. The output classifier version is recorded so this list can be
# revisited when the dependency changes.
KNOWN_LANGID_LANGUAGES = frozenset(
    "af am an ar as az be bg bn br bs ca cs cy da de dz el en eo es et eu fa fi "
    "fo fr ga gl gu he hi hr ht hu hy id is it ja jv ka kk km kn ko ku ky la lb "
    "lo lt lv mg mk ml mn mr ms mt nb ne nl nn no oc or pa pl ps pt qu ro ru rw "
    "se si sk sl sq sr sv sw ta te th tl tr ug uk ur vi vo wa xh zh zu".split()
)


def normalize_language_code(value: Any) -> str | None:
    """Normalize only recognized langid labels and the English/Hindi aliases."""
    if not isinstance(value, str):
        return None
    code = value.strip().lower().replace("_", "-")
    if code in {"mixed", "uncertain"}:
        return code
    code = LANGUAGE_ALIASES.get(code, code)
    return code if code in KNOWN_LANGID_LANGUAGES else None


@dataclass(frozen=True)
class LanguagePolicy:
    language_gate_approved: bool = False
    classifier: str = "langid"
    decision_mode: str = "label_based"
    validation_reference: str | None = None
    approved_by: str | None = None
    minimum_classifier_score_by_field: Mapping[str, float | None] | None = None

    def __post_init__(self) -> None:
        scores = self.minimum_classifier_score_by_field or {}
        if not isinstance(self.language_gate_approved, bool):
            raise ValueError("language_gate_approved must be a JSON boolean")
        if self.classifier != "langid":
            raise ValueError("Only the existing langid classifier is supported")
        if self.decision_mode not in {"label_based", "score_floor"}:
            raise ValueError("decision_mode must be 'label_based' or 'score_floor'")
        if self.language_gate_approved:
            if (
                not isinstance(self.validation_reference, str)
                or not self.validation_reference.strip()
                or not isinstance(self.approved_by, str)
                or not self.approved_by.strip()
            ):
                raise ValueError(
                    "An approved language policy requires validation_reference and approved_by"
                )
            if self.decision_mode == "label_based":
                if any(scores.get(field) is not None for field in ("question", "response")):
                    raise ValueError("label_based mode must not configure classifier-score floors")
            else:
                for field in ("question", "response"):
                    value = scores.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(
                            f"score_floor mode requires a finite raw classifier-score floor for {field}"
                        )

    @property
    def is_operational(self) -> bool:
        if not self.language_gate_approved or not self.validation_reference or not self.approved_by:
            return False
        if self.decision_mode == "label_based":
            return True
        scores = self.minimum_classifier_score_by_field or {}
        return all(scores.get(field) is not None for field in ("question", "response"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LanguagePolicy":
        allowed = {
            "language_gate_approved",
            "classifier",
            "decision_mode",
            "validation_reference",
            "approved_by",
            "minimum_classifier_score_by_field",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown language policy keys: {sorted(unknown)}")
        approved = value.get("language_gate_approved", False)
        if not isinstance(approved, bool):
            raise ValueError("language_gate_approved must be a JSON boolean")
        return cls(
            language_gate_approved=approved,
            classifier=str(value.get("classifier", "langid")),
            decision_mode=str(value.get("decision_mode", "label_based")),
            validation_reference=value.get("validation_reference"),
            approved_by=value.get("approved_by"),
            minimum_classifier_score_by_field=value.get(
                "minimum_classifier_score_by_field", {}
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "LanguagePolicy":
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Language policy file must contain a JSON object")
        return cls.from_dict(document)


def _field_decision(
    field: str,
    prediction: Mapping[str, Any] | None,
    policy: LanguagePolicy,
    verified_mixed_components: Mapping[str, Sequence[str]],
    mixed_review_reference: str | None,
) -> tuple[str, str | None]:
    if not isinstance(prediction, Mapping):
        return "uncertain", "LANGUAGE_METADATA_MISSING_OR_INVALID"
    if prediction.get("status") != "classified":
        return "uncertain", "LANGUAGE_METADATA_MISSING_OR_INVALID"
    if prediction.get("score_is_probability") is not False:
        return "uncertain", "LANGUAGE_METADATA_MISSING_OR_INVALID"

    code = normalize_language_code(prediction.get("language"))
    if code is None:
        return "uncertain", "UNKNOWN_OR_INVALID_LANGUAGE_CODE"
    if code == "uncertain":
        return "uncertain", "LANGUAGE_UNCERTAIN"

    if code == "mixed":
        return "mixed_unverified", "UNVERIFIED_MIXED_LANGUAGE"

    if policy.decision_mode == "score_floor":
        score = prediction.get("classifier_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return "uncertain", "LANGUAGE_METADATA_MISSING_OR_INVALID"
        if not math.isfinite(float(score)):
            return "uncertain", "LANGUAGE_METADATA_MISSING_OR_INVALID"
        threshold = (policy.minimum_classifier_score_by_field or {}).get(field)
        if threshold is None:
            return "uncertain", "LANGUAGE_GATE_NOT_APPROVED"
        if float(score) < float(threshold):
            return "uncertain", "LOW_CLASSIFIER_SCORE"

    if code in SUPPORTED_LANGUAGES:
        return code, None
    return "unsupported", f"UNSUPPORTED_{field.upper()}_LANGUAGE"


def decide_language_outcome(
    question_prediction: Mapping[str, Any] | None,
    response_prediction: Mapping[str, Any] | None,
    policy: LanguagePolicy,
    *,
    verified_mixed_components: Mapping[str, Sequence[str]] | None = None,
    mixed_review_reference: str | None = None,
) -> dict[str, Any]:
    """Return one mutually exclusive language outcome and auditable reasons.

    Only same-language English-English or Hindi-Hindi pairs pass. Cross-language
    pairs, unsupported labels, mixed/uncertain labels, and missing metadata are
    quarantined. Language predictions alone never cause REJECT.
    """
    predictions = {
        "question": question_prediction,
        "response": response_prediction,
    }
    secondary: list[str] = []
    for field, prediction in predictions.items():
        if isinstance(prediction, Mapping):
            code = normalize_language_code(prediction.get("language"))
            if code and code not in (*SUPPORTED_LANGUAGES, "mixed", "uncertain"):
                secondary.append(f"PREDICTED_{field.upper()}_{code.upper()}")
            elif code == "mixed":
                secondary.append(f"PREDICTED_{field.upper()}_MIXED")
            if prediction.get("status") != "classified":
                secondary.append("LANGUAGE_METADATA_MISSING_OR_INVALID")
            elif prediction.get("score_is_probability") is not False:
                secondary.append("LANGUAGE_METADATA_MISSING_OR_INVALID")
            elif code is None:
                secondary.append("UNKNOWN_OR_INVALID_LANGUAGE_CODE")
            elif code == "mixed":
                secondary.append("UNVERIFIED_MIXED_LANGUAGE")
            elif code == "uncertain":
                secondary.append("LANGUAGE_UNCERTAIN")
            score = prediction.get("classifier_score")
            if policy.decision_mode == "score_floor" and (
                prediction.get("status") == "classified"
                and (
                    prediction.get("score_is_probability") is not False
                    or isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                )
            ):
                secondary.append("LANGUAGE_METADATA_MISSING_OR_INVALID")
        else:
            secondary.append("LANGUAGE_METADATA_MISSING_OR_INVALID")

    # No language prediction can drive an automated outcome until the full
    # validation/approval gate has been recorded in configuration.
    if not policy.is_operational:
        return {
            "outcome": "QUARANTINE",
            "reason_code": "LANGUAGE_GATE_NOT_APPROVED",
            "reason": "No approved operational language policy is configured.",
            "secondary_reason_codes": sorted(set(secondary)),
            "language_gate_operational": False,
        }

    components = verified_mixed_components or {}
    q_state, q_reason = _field_decision(
        "question", question_prediction, policy, components, mixed_review_reference
    )
    r_state, r_reason = _field_decision(
        "response", response_prediction, policy, components, mixed_review_reference
    )
    field_reasons = [reason for reason in (q_reason, r_reason) if reason]
    unsupported = [field for field, state in (("question", q_state), ("response", r_state)) if state == "unsupported"]
    if unsupported:
        secondary.extend(field_reasons)
        return {
            "outcome": "QUARANTINE",
            "reason_code": "UNSUPPORTED_LANGUAGE_REVIEW",
            "reason": "At least one field is predicted as a language outside the English/Hindi V1 scope; language predictions alone never cause REJECT.",
            "secondary_reason_codes": sorted(set(secondary)),
            "language_gate_operational": True,
        }

    unresolved = {"uncertain", "mixed_unverified"}
    if q_state in unresolved or r_state in unresolved:
        reason_code = (
            "UNVERIFIED_MIXED_LANGUAGE"
            if "mixed_unverified" in {q_state, r_state}
            else next(
                (
                    reason
                    for reason in field_reasons
                    if reason in {
                        "LANGUAGE_METADATA_MISSING_OR_INVALID",
                        "UNKNOWN_OR_INVALID_LANGUAGE_CODE",
                        "LOW_CLASSIFIER_SCORE",
                    }
                ),
                "LANGUAGE_UNCERTAIN",
            )
        )
        secondary.extend(field_reasons)
        return {
            "outcome": "QUARANTINE",
            "reason_code": reason_code,
            "reason": "At least one field lacks a validated, in-scope language decision.",
            "secondary_reason_codes": sorted(set(secondary)),
            "language_gate_operational": True,
        }

    if {q_state, r_state} == {"en", "hi"}:
        return {
            "outcome": "QUARANTINE",
            "reason_code": "CROSS_LANGUAGE_PAIR_REVIEW",
            "reason": "English/Hindi cross-language question-answer pairs are held for separate review in V1.",
            "secondary_reason_codes": sorted(set(secondary)),
            "language_gate_operational": True,
        }

    if q_state == r_state == "en":
        reason_code = "SUPPORTED_EN_EN"
    elif q_state == r_state == "hi":
        reason_code = "SUPPORTED_HI_HI"
    else:  # Defensive: never turn an unrecognized state into PASS.
        return {
            "outcome": "QUARANTINE",
            "reason_code": "LANGUAGE_UNCERTAIN",
            "reason": "The field language combination is not covered by an approved PASS rule.",
            "secondary_reason_codes": sorted(set(secondary)),
            "language_gate_operational": True,
        }

    return {
        "outcome": "PASS",
        "reason_code": reason_code,
        "reason": "Both fields have the same supported English/Hindi label under the approved label-based policy.",
        "secondary_reason_codes": sorted(set(secondary)),
        "language_gate_operational": True,
    }


def build_language_outcome_record(
    *,
    dataset_revision: str,
    source_shard: str,
    source_row_number_1_based: int,
    question: Any,
    response: Any,
    question_prediction: Mapping[str, Any],
    response_prediction: Mapping[str, Any],
    decision: Mapping[str, Any],
    classifier_version: str,
    sampling_metadata: Mapping[str, Any] | None = None,
    quality_signals: Mapping[str, Any] | None = None,
    context_sha256: str | None = None,
) -> dict[str, Any]:
    """Attach source provenance, field predictions, and one final language outcome."""
    source_row_id = (
        f"{dataset_revision}:{source_shard}:{source_row_number_1_based}"
    )
    return {
        "source_row_id": source_row_id,
        "dataset_revision": dataset_revision,
        "source_shard": source_shard,
        "source_row_number_1_based": source_row_number_1_based,
        "question": question,
        "response": response,
        "question_prediction": dict(question_prediction),
        "response_prediction": dict(response_prediction),
        "language_detection_method": "langid_per_field",
        "language_classifier_version": classifier_version,
        "outcome": decision["outcome"],
        "outcome_scope": "language_eligibility_only",
        "reason_code": decision["reason_code"],
        "reason": decision["reason"],
        "secondary_reason_codes": list(decision.get("secondary_reason_codes", [])),
        # Other Stage 3 quality rules remain independent from language eligibility.
        "quality_signals": dict(quality_signals or {}),
        "context_sha256": context_sha256,
        "language_gate_operational": bool(
            decision.get("language_gate_operational", False)
        ),
        "sampling_metadata": dict(sampling_metadata or {"selection_type": "full_scan"}),
        "context_use": "provenance_only; not used for language classification",
    }
