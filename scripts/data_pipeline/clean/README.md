# Stage 3 cleaning

`run_language_clean.py` runs the bounded-memory Stage 3 decision pass over the
raw Parquet shards. It reuses the audit's deterministic flags and NFC/whitespace
record fingerprint. It never edits raw files, creates Stage 4 splits, or writes
MLX-LM training JSONL.

## Approved V1 language policy

The explicit operational config is
[`language_policy.label_based_v1.json`](language_policy.label_based_v1.json).
It passes only English-English or Hindi-Hindi field-label pairs. Cross-language
pairs, unsupported predictions, mixed/uncertain labels, and missing/invalid
predictions are quarantined. A language prediction never causes REJECT. Raw
`langid` scores are preserved for analysis and are not compared to thresholds.
This is an operational classifier rule, not a calibrated guarantee of language
identity. The default template remains disabled to keep accidental runs closed.

## Smoke run

```bash
uv run python scripts/data_pipeline/clean/run_language_clean.py \
  --policy-config scripts/data_pipeline/clean/language_policy.label_based_v1.json \
  --sample-per-shard 30 \
  --seed 20261029 \
  --batch-size 2048 \
  --output-dir data/reports/stage3-smoke-label-v1
```

The fixed-seed reservoir selects up to 30 rows per shard without preferring
early rows. It reads the source fields in Arrow batches, classifies only the
selected question and response separately, and uses context only for audit
signals and provenance fingerprints. Smoke outcomes are not corpus estimates.

## Full Stage 3 pass

After reviewing the smoke outputs, run:

```bash
uv run python scripts/data_pipeline/clean/run_language_clean.py \
  --policy-config scripts/data_pipeline/clean/language_policy.label_based_v1.json \
  --full-scan \
  --batch-size 2048 \
  --output-dir data/processed/stage3-label-based-v1
```

The run refuses to overwrite a nonempty output directory and checks available
disk space. It streams one bounded Arrow batch at a time and uses a disk-backed
SQLite fingerprint index with a bounded cache. Sorted shard names and ascending
source rows define the canonical duplicate record.

The output contains:

- `outcome_partitions/PASS/`, `QUARANTINE/`, and `REJECT/`: Parquet parts with
  unchanged question/response text, source row IDs, per-field language labels
  and raw scores, quality flags, reason codes, text hashes, and duplicate links.
- `outcome_ledger.jsonl.gz`: one compact provenance/reason record per source
  row; it omits question/response text because those are in the Parquet parts.
- `fingerprint_index.sqlite3`: disk-backed normalized full-record fingerprints
  and deterministic canonical IDs.
- `stage3_summary.json` and `stage3_report.md`: outcome and reason counts,
  language observations, duplicates, integrity checks, representative records,
  source checksums, output reconciliation, and memory/disk details.
- `source_manifest_before.json` and `source_manifest_after.json`, plus the exact
  `language_policy_used.json` config and its checksum.

Null/blank questions or responses and redundant exact full-record occurrences
are REJECTED under the approved deterministic rules. Exact placeholder-response
matches, context-dependent or statement-like prompts, question/response text
integrity flags, material-field missing-reference matches, unsupported or
unresolved language predictions, and long responses are QUARANTINED for review.
Short responses and context-only integrity/reference signals are retained as
review signals and do not independently block PASS. Form-feed page boundaries
are preserved. No questions or answers are repaired or normalized.

Stage 4 should consume only the Stage 3 PASS Parquet pool and perform its own
grouped splits, tokenizer checks, final holdout allocation, and export
validation. PASS is not legal correctness or per-record semantic-answerability
certification; this Stage 3 pass does not adjudicate every response against
authoritative source material.
