# Data Pipeline Deep Dive

[Project overview](../README.md) · [Exact cleaning policy](../scripts/data_pipeline/clean/CLEANING_POLICY.md) · [Stage 3 implementation](../scripts/data_pipeline/clean/run_language_clean.py) · [Stage 4 implementation](../scripts/data_pipeline/export/run_stage4_export.py)

The central engineering contribution is an **immutable-input, auditable, bounded-memory workflow** that transformed **6,055,371 source Q&A records across 33 shards** into **5,726,805 policy-eligible examples** for structured ML training.

## 0. Raw dataset contract

Source: [Prarabdha/indian-legal-supervised-fine-tuning-data](https://huggingface.co/datasets/Prarabdha/indian-legal-supervised-fine-tuning-data), pinned revision `fe58e87d8393ddd0b8e4224e3eec086c37f1f78e`.

Input Parquet shards have three relevant fields: `context`, `question` and `response`. `context` is **not included in model training messages**. A source-row identifier preserves snapshot, shard and 1-based row information. The raw dataset is read-only. The upstream content is not automatically verified against primary statutes or court documents.

## 1. Audit before cleaning

Implementation: [`run_audit.py`](../scripts/data_pipeline/audit/run_audit.py).

The audit scans all 33 shards, reconciles Arrow metadata with observed row counts, computes field-type/length distributions, and emits diagnostic flags without modifying or rejecting any source record.

| Signal | Count | Interpretation |
|---|---:|---|
| Missing questions | 17,123 | A required field is unavailable |
| Suspected context-dependent questions | 214,273 | The prompt may refer to an unseen judgment or passage |
| Statement-like questions | 24,331 | Possibly unsuitable as instruction examples |
| Short responses (<20 characters) | 146,921 | **Not** necessarily wrong; a short amount/date may be valid |
| Placeholder response patterns | 11,459 | Limited useful supervision may remain |
| Possible missing reference numbers | 422,913 | Heuristic matches, many inside source `context` |
| Redundant normalized full-record occurrences | 2,990 | Canonical deduplication candidates |
| Suspicious control characters | Context 39,961; Q 17; A 83 | Review text integrity |
| Replacement characters (`U+FFFD`) | Context 135; Q 13; A 70 | Possible irreversible encoding damage |
| Form-feed (`U+000C`) | Context 62,329 occurrences | Often a **page break**, not automatically corruption |

Flags overlap, so summing them **does not** yield a count of unique defective rows. The audit does not prove semantic answerability or legal correctness. Representative examples, exact rules and limitations are documented in [`CLEANING_POLICY.md`](../scripts/data_pipeline/clean/CLEANING_POLICY.md).

### Normalization and fingerprints

For comparing records, text is normalized using **Unicode NFC and collapsed whitespace**. The original text is preserved. Stable SHA-256 fingerprints support canonical duplicate detection; the rule does not make aggressive case-folding, punctuation stripping or semantic-near-duplicate claims. Canonical order follows sorted shard names and ascending source row numbers.

## 2. Language detection and validation

The pipeline tests the language of `question` and `response` **independently**, using the `langid` classifier. A separate review process examined 10K candidate pairs and a **500-pair pilot** (400 probability-sampled core examples and 100 targeted edge cases). The completed pilot labels were **AI-assisted and user-confirmed**, not independently double-annotated.

Key finding: short text, Romanized Hindi, unusual names and legal citations were sometimes misclassified. The classifier's scores are unnormalized—**not probabilities**—and the reviewed pilot could not justify reliable production score floors. Rather than guess thresholds, the approved operational rule is conservative:

| Question/response language prediction | Outcome from language gate |
|---|---|
| Same supported English/English or Hindi/Hindi | Eligible to PASS subsequent quality checks |
| English/Hindi cross-language pair | QUARANTINE |
| Unsupported, mixed-labelled, missing, uncertain or invalid label | QUARANTINE |
| Unsupported language prediction alone | **Never** automatic REJECT |

The classifier does not reliably detect all mixed-language text. The gate is a measurable *operational compromise*, not a guarantee of language correctness. It is intentionally distinct from the text-quality and legally answerable-question checks. See the [language policy implementation](../scripts/data_pipeline/clean/language_policy.py) and [policy configuration](../scripts/data_pipeline/clean/language_policy.label_based_v1.json).

## 3. Stage 3: streaming data decisions

Implementation: [`run_language_clean.py`](../scripts/data_pipeline/clean/run_language_clean.py). All source rows receive a machine-readable primary outcome and all applicable secondary reason codes.

```text
Parquet batch
   ├─ validate input fields; compute source text hashes
   ├─ audit lexical, control-character, and quality indicators
   ├─ classify question and answer languages separately
   ├─ query SQLite normalized-record fingerprint index
   └─ apply deterministic decision precedence
         ├─ REJECT: null/blank required field or redundant full-record duplicate
         ├─ QUARANTINE: unresolved language, placeholder, source-dependence,
         │              material-field reference/integrity concern, etc.
         └─ PASS: no blocking deterministic rule or review signal
               ↓
Outcome-partitioned Parquet + provenance ledger + source manifests + reports
```

**Precedence:** `REJECT` before `QUARANTINE` before `PASS`. A row has one final outcome even when it triggers several review signals. A positive PASS is **not a legal correctness certificate**.

**Why PyArrow + SQLite?** Millions of fingerprints would use substantial memory if retained as an in-memory Python set. Bounded Arrow batches and disk-backed SQLite permitted the entire run with **~637 MiB peak RSS** while preserving global duplicate consistency.

**Why no automatic text repair?** Inserting a missing legal section number, reconstructing a damaged name or altering source answers risks inventing legal facts. For that reason, transformed outputs keep source question/response text intact, using hashes to verify no unexpected mutations.

### Reconciled outcomes

| Final outcome | Rows | Share |
|---|---:|---:|
| PASS | 5,726,805 | 94.5740% |
| QUARANTINE | 309,434 | 5.1101% |
| REJECT | 19,132 | 0.3160% |
| **TOTAL** | **6,055,371** | **100%** |

Output artifacts in the local Git-ignored `data/processed/stage3-label-based-v1/` include:

- `outcome_partitions/{PASS,QUARANTINE,REJECT}/` — partitioned Parquet records and reason metadata.
- `outcome_ledger.jsonl.gz` — compact one-row-per-input decision/provenance ledger.
- `fingerprint_index.sqlite3` — on-disk canonical duplicate index.
- `stage3_report.md`, `stage3_summary.json` — reconciliations and sampled examples.
- Before/after source SHA-256 manifests and exact applied language-policy config.

The full scan verified **33/33 shards**, exact input/outcome row counts, matching raw source hashes, and **zero unexpected question/response text hash mismatches**.

## 4. Multiple-response question audit

The Stage 3 PASS pool contained **5,690,343 normalized question groups**. A follow-up exact-question analysis found **17,967 groups with multiple distinct answers**, containing **49,052 records** and **46,065 response variants**.

Different answers might be legally compatible, paraphrases, context-specific or contradictory; the audit does not infer which. It generated complete compressed group data plus a deterministic 100-group review sheet. Stage 4 preserved variants and group integrity instead of guessing one “correct” answer.

## 5. Stage 4: split, tokenize, validate, export

Implementation: [`run_stage4_export.py`](../scripts/data_pipeline/export/run_stage4_export.py).

**Leakage protection:** a deterministic SHA-256 key groups **normalized question text**, then assigns the whole group to one split (seed `20261031`); no answer variant from the same normalized question can cross that boundary.

| Split | Records | Groups | Multiple-response groups |
|---|---:|---:|---:|
| Train | 5,154,546 | 5,121,629 | 16,140 |
| Validation | 286,251 | 284,432 | 919 |
| Test | 286,008 | 284,282 | 908 |

Output chat records have **only** a question and answer:

```json
{"messages": [{"role": "user", "content": "<question>"}, {"role": "assistant", "content": "<answer>"}]}
```

The exporter applies the real Qwen3-4B tokenizer and chat template for sequence-length checks. Across splits, **P99 was 229 tokens**, while the recommended initial train maximum was **1,024**. **112 examples** exceeded that recommendation (maximum 1,291); they remained untruncated in Stage 4 for explicit later handling. No record exceeded Qwen3-4B's configured **40,960** tokens.

Checks reconciled all JSONL rows to Stage 3 PASS, verified roles/schema and unique source IDs, preserved provenance through a manifest, and measured **zero normalized question-group overlaps across splits**. This does not rule out *semantic paraphrase* leakage between differently worded questions.

## Open limitations

- `langid` output is a prediction, not a validated Hindi/English truth label for every row.
- Exact normalization does not resolve semantically similar or contradictory legal questions.
- Some Q&A examples may require source context that the inference prompt deliberately omits.
- PASS signals compliance with project cleaning policy, **not legal correctness**.
- The final 30-question test sample has since been used for evaluation and should not be reused as a never-seen holdout for future checkpoint decisions.

For reproduce-safe commands, see [Reproducibility](REPRODUCIBILITY.md).
