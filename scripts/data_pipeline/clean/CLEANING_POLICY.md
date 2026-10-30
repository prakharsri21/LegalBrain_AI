# LegalBrain AI Stage 3 Cleaning Policy

**Status:** Stage 3 implementation and the full-corpus pass for this revision are complete. The user-approved V1 language gate is label-based, uses no score floors, and language predictions never cause REJECT. The classifier is imperfect and this policy is not a calibrated guarantee of language identity. Review signals remain heuristic; Stage 4 splits and exports are outside this policy.

**Dataset revision:** `fe58e87d8393ddd0b8e4224e3eec086c37f1f78e`

**Audit:** 6,055,371 rows across 33 shards; all shard metadata counts reconcile.
**Version 1 target scope:** Same-language English-English or Hindi-Hindi question-to-answer pairs. Cross-language and mixed/uncertain fields remain quarantined. The source `context` is read only for audit signals and duplicate fingerprints; Stage 3 stores a context hash and source-row pointer, not raw context text. Context must not be part of the model prompt or training input.

**Stage boundary:** Stage 3 owns record-level outcomes, approved cleaning transformations, provenance, and input-to-outcome reconciliation. Stage 3 does not create train/validation/test splits, perform tokenizer checks, serialize final MLX-LM JSONL, or validate exports. Stage 4 owns grouped split assignment, final holdout allocation, tokenizer checks, JSONL serialization, and export validation.

## 1. Executive summary

The original Parquet dataset is immutable. Stage 3 should read it in bounded batches and write a derived record-level dataset, a row-level decision ledger, and reconciliation reports. It must never edit or delete raw data. Stage 4 consumes Stage 3's PASS pool and owns all split and final-export work.

Every input row must receive exactly one mutually exclusive outcome:

- **PASS:** The row satisfies approved deterministic Stage 3 rules and has no unresolved required-review trigger. PASS means eligible for Stage 4 processing; it does **not** certify legal correctness or prove semantic answerability for that individual record. It does not imply that every row received full semantic or legal review.
- **QUARANTINE:** The row remains potentially useful but has unresolved language, question-only suitability, source-support, target-conflict, text-integrity, or review issues. Do not include it in the primary training export unless the specified review clears it.
- **REJECT:** The row is excluded from the derived V1 training data by an approved, explicit deterministic rule. Keep the original row, source location, reason, and any canonical-record link.

Use deterministic Stage 3 precedence: (1) reject null/blank questions or responses and redundant normalized exact full-record duplicate occurrences; (2) otherwise quarantine if a language or quality review trigger remains unresolved; (3) otherwise pass only if all approved deterministic checks succeed. Language predictions never cause REJECT, including unsupported-language predictions. Store one primary outcome and primary reason plus all applicable secondary reason codes. This preserves one outcome per row without discarding overlapping evidence.

The implemented deterministic REJECT rules are null/blank questions or responses and redundant normalized exact full-record duplicate occurrences. Exact placeholder responses remain QUARANTINE because the finite templates are not approved as hard exclusions. Unsupported, mixed, cross-language, missing, or uncertain language predictions are also QUARANTINE. Short responses, form-feeds, missing-reference regex matches, and statement-like prompts are not automatic rejection rules.

## 2. Evidence from the completed audit

Values below are observed in `data/reports/audit-full/audit_summary.json`. Individual flags overlap and must not be added to estimate unique affected rows.

### Coverage and field counts

- **6,055,371** rows were audited; the sum of the 33 shard metadata counts is **6,055,371** and every shard is marked fully scanned.
- `context`: 6,055,371 strings; 0 null, non-string, or blank; mean length 2,254.21 characters (min 1, max 2,500).
- `question`: 6,038,248 strings; **17,123 nulls**; 0 non-string or blank; mean length 127.38 (min 10, max 1,384).
- `response`: 6,055,371 strings; 0 null, non-string, or blank; mean length 285.12 (min 1, max 10,997).

### Main review signals

- `missing_question`: **17,123**, matching the null question count. No blank question strings were observed.
- `placeholder_response`: **11,459**. In the capped 20-row flagged sample, 19 examples also had `missing_question`; this is sample overlap, not a full-corpus cross-tab. The summary does not report the population overlap or response-template breakdown.
- `short_response`: **146,921**, using the audit's `<20` character threshold. Sample responses include `17 March, 2008`, `Rs 500`, `4,67,39,503`, and `b. Nil`; brevity alone does not make these unusable.
- `long_response`: **2**, using the audit's `>10,000` character threshold. A sample begins a Markdown table and warrants a truncation/format check.
- `possible_context_dependent_question`: **214,273**. This lexical question-field heuristic includes phrases such as “in this case,” “the above,” and “the impugned order.” Some sampled prompts name a case/provision or request a date/amount and may still be answerable without the source context.
- `statement_like_question`: **24,331**. Flagged examples include declarative propositions paired with `Yes.` or a restatement. Random review examples also contain declarative-looking, unflagged questions, such as “The trial court's judgment on the pleadings and evidence is valid for a title and possession suit.” The detector is incomplete, and absence of `?` is not sufficient evidence for rejection.
- `missing_legal_reference_number`: **422,913** across `context`, `question`, and `response`. Capped examples include “Under clause   of the Ordinance” and “under Section   of the Negotiable Instruments Act.” These look like missing citation identifiers, but this flag count is **not** a confirmed defect count. The report does not include a population count by matched field or a validated precision estimate. A random example with “Section 83” was not flagged.
- Suspicious control-character flags: context **39,961**, question **17**, response **83**. Samples include U+0005, U+0013/U+0014 around “C Form,” and U+0016 in an address; the surrounding text is needed to determine whether meaning changed.
- U+FFFD replacement-character flags: context **135**, question **13**, response **70**. Samples include `Honb�'ble`, a damaged Hindi word, and multilingual response text with repeated replacement characters. Unicode normalization cannot reconstruct lost characters.
- Form-feed U+000C is counted separately: **62,329 occurrences** across 47,596 contexts, 5 questions, and 4 responses. A sampled context uses `\f` as a page boundary and has no audit flags.

### Duplicate statistics

Audit fingerprints use NFC Unicode normalization and whitespace collapsing; they do not lowercase text or remove punctuation.

- Unique normalized full records: **6,052,381**; redundant full-record duplicate occurrences: **2,990**.
- Unique nonblank questions with nonblank answers: **5,988,272**.
- Unique normalized question-response pairs: **6,026,477**; extra occurrences of repeated pairs: **11,771**.
- Questions with multiple distinct responses: **21,773**.

These are aggregate statistics. The report does not include duplicate-group examples or a cross-tab showing how pair repetitions relate to different contexts. It cannot tell us how many multiple-response groups are paraphrases, context-specific answers, or real target conflicts. Near-duplicate detection (including MinHash/LSH) was not implemented or measured.

### Language estimate

The audit sampled 5,000 question-response texts, classified 4,993, and skipped 7 short texts. The sample estimate was English 4,979, Chinese 7, Hindi 4, and Latin 3. This is not a full-corpus census. Short legal language, names, quotations, and mixed-language text can be misclassified.

## 3. Decision model and deterministic precedence

The following sequence assigns one outcome to every source row. If multiple rules apply, retain all reasons but assign the outcome from the first applicable tier.

1. **REJECT** a null/blank question or response, or a redundant normalized exact full-record duplicate occurrence. The earliest row in sorted shard-name and 1-based-row order is the canonical record. Exact placeholder responses remain **QUARANTINE** until an approved finite template list authorizes exclusion. Language predictions never cause REJECT.
2. **QUARANTINE** if no REJECT rule applies and a language or quality review trigger remains unresolved. Keep every QUARANTINE row out of the primary Stage 4 training pool unless review clears it. Heuristic signals remain distinct from confirmed defects.
3. **PASS** only if both fields have the same supported language label (`en`/`eng` with `en`/`eng`, or `hi`/`hin` with `hi`/`hin`), no unresolved mixed/unsupported language signal exists, and no approved quality review trigger remains. PASS indicates policy eligibility, not a legal or semantic certification.

The decision record must contain `source_row_id`, dataset revision, source shard, 1-based source row, exactly one outcome, one primary reason code, all secondary reason/signal codes, and any canonical duplicate ID or review disposition. For PASS, record a positive reason such as `approved_checks_passed`; do not represent PASS only by an empty flag list.

### PASS gate

For V1, a row can pass the deterministic gate only if question and response are nonblank strings, the approved English/Hindi language gate permits it, no approved REJECT rule applies, and no unresolved required-review trigger remains. Apply approved deterministic checks and conservative, validated heuristics; do not imply that every PASS row received individual semantic or legal review. If the language gate is not approved, do not treat the audit sample as sufficient to pass or reject language-dependent rows for the English/Hindi pool. PASS is eligibility under this policy, not a finding of legal correctness or proof of full semantic answerability.

## 4. Proposed rule policy

Counts are observed flag counts, not mutually exclusive row counts. Rules marked review-only must remain so until the review and approval are complete.

### A. Null or blank question/response

- **Condition / fields:** `question` or `response` is null or blank after safe whitespace normalization.
- **Outcome:** **REJECT** from V1 training data.
- **Affected:** 17,123 null questions; 0 blank questions; 0 null, blank, or non-string responses in this audit.
- **Rationale and risk:** A supervised pair needs both an input and target. The false-positive risk is low for a genuinely null/blank field. Do not synthesize text from context under this policy.
- **Automation:** Applied as an approved deterministic Stage 3 exclusion in this run. Preserve the raw row and reason in the rejected partition and ledger.

### B. Confirmed exact placeholder response

- **Condition / field:** The entire normalized `response` exactly matches an approved finite list of generic fallback templates, such as `The answer is not available.`
- **Outcome:** **QUARANTINE** exact matches for review in this run. The finite matching list is not approved as a hard-exclusion list, so these rows are not REJECTED.
- **Affected:** 11,459 `placeholder_response` flags. The capped flagged sample shows substantial overlap with missing questions, but no exact corpus overlap is reported.
- **Rationale and risk:** A fallback is not an answer target. Use exact whole-response matching after conservative normalization (Unicode NFC, trim/collapse whitespace, case-fold, and a narrowly specified terminal punctuation policy). Do not use substring matching or broad semantic matching. A short answer such as `Not available` might be legitimate depending on the question; leave ambiguous templates in quarantine.
- **Automation:** Reliable only for the human-approved exact list. Record the matched template ID.

### C. Normalized exact full-record duplicate occurrence

- **Condition / fields:** All three fields match under the audit's NFC-plus-whitespace normalization and the row is a redundant occurrence.
- **Outcome:** **REJECT** that duplicate occurrence from the derived V1 set; keep one deterministic canonical record, which must still pass all other rules.
- **Affected:** 2,990 occurrences; 6,052,381 unique normalized full records.
- **Rationale and risk:** This removes exact repeated training weight while preserving one source record. Normalization deliberately treats whitespace and Unicode composition differences as equal; retain originals and do not extend equality to different questions, answers, or contexts.
- **Automation:** Applied deterministically in this run. The earliest `(shard filename, 1-based row)` is retained as canonical; every later occurrence points to that source row ID.

### D. Repeated question-response pairs and multiple responses

- **Condition / fields:** Normalized question-response pairs repeat, or a normalized question has more than one distinct response; retain context/document provenance for review.
- **Outcome:** Do not reject based on these counts. **QUARANTINE** unresolved multi-response question groups where the question alone cannot distinguish targets. Repeated identical pairs may remain eligible with controlled weighting if provenance and split handling are sound.
- **Affected:** 11,771 extra repeated question-response occurrences; 21,773 questions with multiple distinct responses. These are different metrics and must not be summed.
- **Rationale and risk:** A repeat may be redundant, but context can differ. Multiple responses can be paraphrases, complementary answers, or contradictory case outcomes. The audit report does not classify them.
- **Automation:** Pair/group detection can be deterministic; deciding whether answers conflict or are compatible requires sample and group-level review. Do not automatically choose one answer.

### E. English/Hindi V1 language scope and operational approval

- **Condition / fields:** Classify `question` and `response` independently. Never append `context` to either field or use it to decide language. The existing audit's 4,993 classified texts from an approximately 5,000-text sample are not a census and cannot classify individual rows.
- **Supported scope:** Normalize `en`/`eng` to English and `hi`/`hin` to Hindi. Only codes emitted by the configured classifier's recognized language inventory are treated as known. Unknown or malformed codes are **QUARANTINE**, not assumed supported or rejected.
- **Operational decision logic:** The approved V1 rule passes only English-English or Hindi-Hindi field-label pairs. English-Hindi cross-language pairs, unsupported language labels, `mixed`, `uncertain`, missing/invalid predictions, and all other ambiguous combinations are **QUARANTINE**. The language module has no REJECT outcome. This is a label-based operational gate, not proof that `langid` identified either field correctly.
- **Score handling:** Preserve each raw score as uncalibrated evidence; never compare it to a floor in `label_based` mode. The policy configuration has separate question/response floor fields set to `null`. The classifier's score direction and length dependence were documented in the pilot evaluation; no production floors were recommended.
- **Validation evidence (2026-10-29):** The 500-record reviewed copy is an AI-assisted draft confirmed by one user, not independent multiple-reviewer annotation. It has 500 unique IDs, no missing required labels, no invalid values, and no detected mixed-status consistency issues. The installed classifier (`langid` 1.1.6; `NORM_PROBS=False`) reproduced stored top labels on all 500 question and response fields separately using the Stage 3 helper and 2,000-character cap. The 400-record `uniform_core` group produced question accuracy 396/397 (0.9975; three missing predictions) and response accuracy 392/400 (0.9800). The 100 targeted records produced question accuracy 77/100 and response accuracy 46/100; targeted scores are for failure discovery, not population estimates. Core question labels had no unsupported examples and core response labels had one, so rare-language error rates remain weakly measured. The approximately 5,000-text audit estimate is not a census.
- **Approval status:** The user approved this conservative label-based policy in the Stage 3 task. Full-corpus processing may use `scripts/data_pipeline/clean/language_policy.label_based_v1.json`. The default template remains disabled as a fail-closed fallback. The classifier is imperfect; English/Hindi filtering can underrepresent excluded languages and code-switched legal usage. Keep original predictions, raw scores, uncertainty, and reviewed labels as separate evidence.
- **Mixed language:** All `mixed` predictions remain **QUARANTINE**. A high score or generic mixed label is not constituent-language verification. This V1 policy has no mixed-field PASS path.
- **Selection bias:** English/Hindi filtering can underrepresent excluded languages and code-switched legal usage. Preserve predictions, raw scores, uncertainty, and any human-reviewed labels as separate evidence. Human labels, when available, must never overwrite the original classifier predictions.
- **Automation:** `scripts/data_pipeline/clean/run_language_clean.py` applies Stage 3 language and quality outcomes in bounded batches. It writes PASS/QUARANTINE/REJECT Parquet partitions, a compact outcome ledger, source manifests, a summary, and a disk-backed SQLite fingerprint index. The default language policy fails closed; the approved label-based config is explicit. It does not assign Stage 4 splits or create training JSONL exports.

### F. Question-only suitability

- **Condition / field:** Whether `question` alone supplies an identifiable task and necessary facts for the proposed `response`.
- **Outcome:** **QUARANTINE** for unresolved high-confidence dependencies, such as a question whose answer hinges on an unidentified “above passage,” unidentified order, or unmentioned case-specific fact. Mere occurrence of “in this case” or another audit phrase is a triage signal, not proof. If the prompt names a case/provision and the task is determinate without hidden facts, it may continue to PASS checks.
- **Affected:** 214,273 `possible_context_dependent_question` flags; this is a question-field lexical heuristic.
- **Risk / validation:** It can overflag self-contained named-case questions and miss implicit dependencies without listed phrases. Use conservative deterministic patterns at scale plus a stratified, human-reviewed sample containing flagged, unflagged, and overlapping cases. Keep question-only suitability separate from answer support.
- **Automation:** High-confidence deterministic patterns may trigger quarantine after approval; the broad audit flag alone remains review-only.

### G. Source-answer support

- **Condition / fields:** Whether `response` is supported by the source evidence and does not contradict the relevant source, considered separately from whether the question is answerable without context.
- **Outcome:** **QUARANTINE** when support is unresolved or context is necessary to identify what the target means. **REJECT** only after a human-reviewed, authoritative contradiction or other explicit unusability rule is established. Otherwise continue to PASS checks.
- **Affected:** No aggregate source-support accuracy count was measured by the audit.
- **Risk / method:** Source `context` is evidence for review, not legal ground truth automatically. A source excerpt can be incomplete, OCR-damaged, or itself wrong. Use an authoritative legal source where required. Do not include context in the model prompt.
- **Automation:** Review-only. Do not infer source correctness from agreement between an LLM and context.

### H. Statement-like questions

- **Condition / field:** `question` is a declarative proposition or assertion rather than a clear question/instruction.
- **Outcome:** **QUARANTINE** from the primary question-answer export until separately reviewed. Do not automatically reject every prompt without `?`; an implicit confirmation/proposition task may be intended. Release only under an approved task definition or reviewed transformation.
- **Affected:** 24,331 `statement_like_question` flags.
- **Evidence / risk:** Flagged examples include “It is crucial for a party to disclose all material facts…” with `Yes.`; random review also has unflagged declarative prompts. Current detection has both false-positive and false-negative risk.
- **Automation:** The audit flag may select review candidates; do not use it as a final REJECT decision.

### I. Short and long responses

- **Condition / field:** `response` under 20 characters or over 10,000 characters, using current audit thresholds.
- **Outcome:** Length alone does not cause REJECT. A short answer may PASS if clear and sufficient; **QUARANTINE** only if the answer is ambiguous, context-dependent, or otherwise uncertain. A long answer may be quarantined for truncation/format review.
- **Affected:** 146,921 short-response flags and 2 long-response flags.
- **Evidence / risk:** Short examples include dates, monetary amounts, numbers, and `b. Nil`, showing why automatic rejection would have false positives. One long sample starts a Markdown table; length also misses shorter truncation.
- **Automation:** Review signals only. Do not trim or reject solely by character count.

### J. Missing legal-reference-number matches

- **Condition / fields:** Current regex may match citation gaps/placeholders involving Article, Section, Rule, Regulation, Clause, or subsection in `context`, `question`, or `response`.
- **Outcome:** **QUARANTINE** only when review confirms a plausible gap and that it materially changes the question, target answer, or source support. A context-only gap need not block a self-contained, supported Q/A pair. A regex match by itself must not reject.
- **Affected:** 422,913 flags. This is not a confirmed defect count; the report does not break it down by matched field or establish precision.
- **Evidence / risk:** Capped examples include apparent OCR gaps such as “Under clause   of the Ordinance”; random negatives include numbered references such as “Section 83.” The capped examples are insufficient to estimate false-positive or false-negative rates. Generic mentions, missing OCR text, and meaningful legal identifiers need different treatment.
- **Automation:** Keep as review-only. The cleaner should log matched field, rule/pattern, and a bounded surrounding excerpt. Repair only against an authoritative source, with original text and repair provenance retained.

### K. Control characters, form-feeds, and U+FFFD

- **Condition / fields:** Suspicious control characters or U+FFFD in `question`/`response`; context equivalents are also recorded for source review. Form-feed U+000C is counted separately.
- **Outcome:** **QUARANTINE** a question/response when the character may change meaning. A context-only artifact quarantines only when it prevents source support or indicates broader corruption. Preserve ordinary form-feed page boundaries; never reject for form-feed alone.
- **Affected:** Control-character flags: context 39,961; question 17; response 83. U+FFFD flags: context 135; question 13; response 70. Form-feed occurrences: context 62,329; question 5; response 4.
- **Risk / treatment:** Sampled U+0005, U+0013/U+0014, and U+0016 may be extraction artifacts, but blind deletion can join words or change punctuation. U+FFFD represents lost data; do not attempt reconstruction without an authoritative source. Normalization in a derived copy must be field-specific, logged, reversible from raw source, and checked with before/after counts.
- **Automation:** Preserve form-feed deterministically. Other control mapping and U+FFFD repair remain review-only until mappings and source evidence are approved.

## 5. Review methodology

Use deterministic heuristics to triage the full corpus; do not run an LLM over all six million records. Draw a stratified human-reviewed sample across shards and across flagged, unflagged, and overlapping cases. Include strata for language predictions and confidence, question-only signals, source-support signals, missing-reference matches by field, placeholders, statement-like questions, short answers, and text corruption. Choose sample sizes based on the desired precision and reviewer capacity; do not treat the audit's 20-per-flag samples as population validation.

Record separate labels for:

1. **Question-only suitability:** Can a reader determine the task and necessary facts from `question` alone, without seeing context?
2. **Source-answer support:** Does reliable source evidence support `response` for the question? Review context separately, and use an authoritative source when needed.
3. Language of question and response, including uncertainty/mixed language.
4. Whether a reference gap or text artifact materially changes meaning.
5. Whether multiple targets are compatible, paraphrases, or conflicting.
6. Proposed outcome and reason, with reviewer identity/version and adjudication status.

An existing local LLM may propose labels for this sample only. It may not classify, reject, or repair all records. Human validation is required; neither LLM labels nor the source context alone are legal ground truth. For question-only suitability, evaluate the question without exposing context; evaluate source support in a separate step where context may be shown.

## 6. Deduplication, provenance, and Stage 4 split recommendations

- Give each source row a stable ID containing dataset revision, shard filename, and 1-based row number. Raw context and original field values remain available in the immutable source; Stage 3 records source pointers and a context fingerprint, and stores unchanged question/response text in outcome partitions. Context is never a model prompt.
- For the 2,990 normalized exact full-record duplicate occurrences, retain one deterministic canonical row, for example the earliest `(shard filename, row number)`. Record every duplicate-to-canonical link. The canonical row still goes through all outcome rules.
- Stage 3 records canonical duplicate links and provenance but does not assign splits. Stage 4 owns grouped split assignment and final holdout allocation.
- Stage 4 should keep repeated question-response pairs together in a split where practical; do not assume they are redundant when contexts differ. Keep unresolved multiple-response question groups together and quarantine for investigation rather than selecting an answer automatically.
- Stage 4 should derive source-document/case groups from reliable identifiers where available. Otherwise use conservative context fingerprints and human-reviewed near-duplicate document clusters. Keep each source-document group within one split.
- **Proposed initial Stage 4 development split:** 90% train, 5% validation, 5% test, grouped by document and with normalized exact duplicates/repeated question-response pairs kept together where practical. Percentages apply to the development pool after a separate final holdout is reserved and after any Stage 4 export-only exclusions are reported. Group sizes and conflicts may make exact percentages infeasible; report achieved proportions and do not break groups merely to hit a target.
- Stage 4 should reserve a fresh, independent final holdout before model selection or prompt tuning and keep it unused for both. The earlier 10-question baseline may overlap the larger training corpus; check overlap against source documents, exact questions/pairs, and near-duplicates. Treat it as exploratory only unless disjointness is established, and do not call it an independent final holdout if overlap is possible.
- Near-duplicate detection is not included in the audit. In a later step, MinHash/LSH may generate candidates over normalized question, response, and document/context text. Preserve legal citations, numbers, names, and dates during normalization. Validate candidate clusters on labeled pairs before using them for split grouping or any removal. Do not claim the audit measured near-duplicates.

Stage 4 owns tokenizer checks, final MLX-LM JSONL serialization, and export validation. The training and inference prompt must use `question` as input and `response` as target; do not supply source `context` to the model. Keep provenance, context, Stage 3 outcomes, and review notes in separate fields/artifacts.

## 7. Stage 3 reconciliation results and remaining metrics

### Stage 3 record-level reconciliation

- The completed Stage 3 run assigns exactly one final outcome (`PASS`, `QUARANTINE`, or `REJECT`) and provenance/reason record for every source row.
- Reconciled counts for this revision: `5,726,805 PASS + 309,434 QUARANTINE + 19,132 REJECT = 6,055,371 input rows`. All 33 shards were fully processed. The runner fails if any row is unassigned, assigned twice, or the counts do not reconcile.
- Separate primary outcome reason from all secondary flag/review signals. Never sum overlapping audit flags to estimate unique rows.
- Report per-outcome, per-reason, per-field, and cross-tab counts, especially question/response nulls, placeholder overlap, duplicate occurrences/canonicals, language outcomes, and legal-reference matches by matched field.
- Record metadata row counts, shard list, dataset revision, code/policy version, deterministic seed, and raw-file checksums. Verify the raw Parquet files remain unchanged.
- The run reports source field type, null, blank, length, control, form-feed, and U+FFFD metrics; unchanged question/response text hashes; zero applied text transformations; and before/after raw shard checksums. Context is not copied to the derived partitions.
- The run reports exact duplicate occurrences, duplicate-group count, deterministic canonical links in the ledger, and examples. It does not group repeated question-response pairs or report conflicting-response groups. The earlier audit measured 21,773 questions with multiple distinct responses; these groups remain a required human-review limitation before final Stage 4 export. No near-duplicate detection was performed.
- Report language predictions and uncertainty as separate evidence. This approved V1 policy uses labels only, no score floor, and no language-based REJECT.

### Stage 4 split and export reconciliation

- Stage 4 must consume the Stage 3 PASS pool; QUARANTINE and REJECT rows cannot silently enter the export.
- Give every Stage 3 PASS row exactly one Stage 4 disposition: train, validation, test, final holdout, or an explicitly counted Stage 4 export exclusion. Reconcile `Stage 3 PASS_count = train_count + validation_count + test_count + final_holdout_count + Stage 4 export_exclusion_count`. For example, tokenizer-length failures must have a reason and count; do not silently lose or double-count rows. If exclusions are detected after a split assignment, preserve and report their intended split as separate provenance.
- Report development split proportions against the eligible development pool after final-holdout reservation and export-only exclusions. Record grouping conflicts and any deviation from the proposed 90/5/5.
- Report split-level leakage checks for source-document groups, normalized exact full-record duplicates, repeated question-response pairs, and validated near-duplicate clusters.
- Report tokenizer configuration and token-length failures, JSONL row counts/schema, serialization checks, and checksums for each final export. Verify source context is absent from the model prompt.
- Keep the fresh final holdout unused for model selection, prompt tuning, or other iterative optimization. Report its allocation and integrity separately from development validation/test data.

## 8. Implementation and run checklist

### Stage 3: record outcomes and cleaning (completed for this revision)

- [x] Pin dataset revision and input shard manifest; read Parquet in bounded batches and never write to raw paths. Full-run source hashes match before and after processing.
- [x] Generate a stable source row ID and one outcome/reason record per input row.
- [x] Apply approved deterministic hard exclusions in fixed precedence: redundant normalized exact full-record duplicate occurrences, then null/blank question or response. Exact placeholder matches remain QUARANTINE; no language prediction causes REJECT.
- [x] Preserve one deterministic canonical duplicate and attach duplicate provenance; evaluate canonical rows under the remaining rules.
- [x] Implement per-field question/response predictions, preserve classifier version and raw scores as non-probability evidence, normalize `en`/`eng` and `hi`/`hin`, and fail closed under an unapproved policy.
- [x] Evaluate the classifier separately by field and by probability-core versus targeted sampling group on the 500-record user-confirmed, AI-assisted pilot; preserve original AI predictions and scores; document score semantics, confusion matrices, error cases, and limitations. The pilot evaluation is evidence; the separate user approval in §4.E authorizes this label-based, no-floor policy.
- [x] Implement decision tests for English, Hindi, verified mixed fields, cross-language pairs, unsupported languages, low scores, missing metadata, unknown codes, precedence, provenance, and context exclusion.
- [x] Keep heuristic signals distinct from confirmed errors. Route unresolved statement-like prompts, context-dependence signals, material-field reference matches, and question/response text-integrity signals to QUARANTINE; short answers, form-feeds, and context-only control/reference signals do not independently block PASS.
- [x] Keep question-only suitability separate from source-answer support. Context is excluded from the language classifier and model prompt; this run does not adjudicate all source support or legal correctness.
- [x] Preserve original question/response strings in derived outcomes and record hashes and signal provenance. Preserve form-feeds; do not reconstruct U+FFFD text.
- [x] Enforce deterministic PASS / QUARANTINE / REJECT precedence and assert one final outcome and reason record for every source row.
- [x] Emit PASS, QUARANTINE, and REJECT Parquet partitions, a provenance ledger, and Stage 3 reconciliation artifacts; do not assign splits, run tokenizer checks, serialize training JSONL, or validate MLX-LM exports.
- [x] Reconcile outcomes, overlapping reasons, per-field metrics, language evidence, duplicate handling, output partitions, and raw checksums; fail closed on unexplained count mismatch.
- [x] Validate Stage 3 rules with the unit suite and a deterministic per-shard smoke run before the full corpus run.

Remaining Stage 3 follow-up: repeated question-response and conflicting-response groups were not emitted by this run. The earlier audit measured 21,773 questions with multiple distinct responses. These require group-level review before Stage 4 export; this pass did not claim they are resolved.

### Stage 4: grouped splits and final exports

- [ ] Consume the approved Stage 3 PASS pool and independently reserve the fresh final holdout before model selection or prompt tuning.
- [ ] Assign source-document groups, normalized exact duplicates, and repeated question-response pairs to splits together where practical; investigate conflicting-response groups and report feasibility deviations.
- [ ] Apply the proposed 90/5/5 grouped split to the eligible development pool; report actual counts and proportions.
- [ ] Run tokenizer checks; record every token-length or other export-only exclusion with a reason and count.
- [ ] Serialize question/response-only MLX-LM JSONL, keeping context and provenance outside the prompt; validate schema, serialization, row counts, and checksums.
- [ ] Give every Stage 3 PASS row exactly one Stage 4 disposition and reconcile Stage 3 PASS pool = train + validation + test + final holdout + explicit Stage 4 export exclusions. If exclusions are detected after split assignment, preserve the intended split separately. Do not silently lose or double-count rows.
- [ ] Verify group leakage constraints and confirm the final holdout was not used for model selection, prompt tuning, or iterative optimization.

## 9. Remaining review and approval decisions

1. What human-review evidence would be required before allowing a mixed question or response to pass; V1 currently quarantines every mixed label.
2. Which exact normalized fallback templates, if any, should become hard exclusions rather than remaining quarantined?
3. What review evidence should release statement-like prompts or context-dependent question candidates?
4. What authority/source hierarchy should reviewers use to verify missing legal references, source support, or damaged text?
5. How should repeated identical question-response pairs across different contexts be weighted, and how should the 21,773 questions with multiple distinct responses measured by the earlier audit be reviewed? The count is not a confirmed conflict count.
6. Is a grouped 90/5/5 split feasible after reserving a fresh final holdout? What document identifier and near-duplicate grouping method are approved?
7. Can the earlier 10-question baseline be proven disjoint from training data? If not, should it be retired and replaced with a fresh holdout?
8. What additional human-review sample is needed to estimate limitations of the quality heuristics and label-based language filter by field and category?

## 10. Completion boundary

The approved label-based language gate and Stage 3 record-level outcome pass are implemented and completed for dataset revision `fe58e87d8393ddd0b8e4224e3eec086c37f1f78e`. The full run processed all 6,055,371 rows across 33 shards and verified that raw source hashes were unchanged. Outputs are in `data/processed/stage3-label-based-v1/`; smoke outputs are separately preserved in `data/reports/stage3-smoke-label-v1/`. This does not resolve the remaining question-response conflict groups, validate every heuristic trigger, certify legal correctness or semantic answerability, or perform Stage 4. No train/validation/test split, final JSONL export, or model training was performed. Stage 4 remains responsible for grouped splits, final holdout, tokenizer checks, and final exports.
