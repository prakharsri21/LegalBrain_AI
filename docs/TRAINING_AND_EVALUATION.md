# Training and evaluation

[Overview](../README.md) · [Pipeline](DATA_PIPELINE.md) · [Reproducibility](REPRODUCIBILITY.md)

## Experiment

The Stage 5 preparation utility selected 100,000 normalized question groups from
the Stage 4 train split using seed 42 and disk-backed SQLite. Five chunks each
contain 20,000 groups, keeping all answer variants together. The fixed validation
sample contains 1,000 Stage 4 validation groups and 1,015 records. The Stage 4
test split was not used to select or train the chunks.

Only chunk 1 was trained: 20,000 groups, 20,113 records, one pass and 20,113
updates. The original Qwen3-4B base started the run, with MLX-LM 0.32.2, LoRA
(rank 8, one layer), Adam, batch size 1, learning rate 5e-6, 1,024-token maximum,
prompt masking, gradient checkpointing and seed 42. The maximum chunk-1 sequence
was 774 tokens. Checkpoints were saved every 1,000 updates; full validation was
run at the first and last steps and every 1,000 steps.

Initial validation token loss was **3.681** before the first optimizer update;
the final adapter independently scored **1.6221** on the same validation sample.
This is a 55.9% reduction in the measured token loss, not proof of legal accuracy.
The sample was repeatedly used for validation, so it is not a fresh final holdout.
The local training report and raw log are ignored artifacts under `outputs/`.

The remaining four chunks were prepared but not trained within the available
compute budget on the 16 GB MacBook Air M4. Five records in those chunks exceed
the 1,024-token training limit and require explicit handling before any future
run. The project does not claim more training would improve factual accuracy.
MLX-LM's adapter resume restores adapter weights, not Adam optimizer state, so a
new invocation is not an exact continuation of the prior optimization run.

## Evaluations

An earlier blind 100-question comparison concerned a separate 50-step pilot.
AI-assisted scores had small differences with paired bootstrap intervals
including zero; 54 of 100 preferences were `neither`. Human scoring fields were
blank. It did not establish a response-quality advantage.

The final comparison used 30 distinct normalized groups selected from the Stage 4
test file by seed-42 SHA-256 rank, retaining the first source record in each
selected group as its reference. The streaming selection used bounded memory;
`scripts/evaluate_model.py` then evaluated the small sample twice using the
Qwen chat template, `/no_think`, greedy decoding and a 256-token output limit.
The outputs are local ignored artifacts under `outputs/evaluations/stage5_final_test_seed42/`.

| Measure | Base | Chunk-1 adapter |
|---|---:|---:|
| Mean unigram overlap F1 against one reference | 0.223 | 0.405 |
| Exact reference match | 0/30 | 1/30 |
| Median output length in word tokens | 172 | 53 |

The test sample is small, contains potentially context-dependent questions and
has not been independently assessed for legal correctness. Lexical metrics
reward similar wording and cannot establish semantic accuracy. Preliminary
AI-assisted triage flagged possible invented dates, case details and incomplete
answers for human verification; it is not a legal adjudication. Finish reasons
were not recorded, so the number of truncated answers is unknown. The 30 test
groups have now been used; reserve a fresh independent holdout for future final
assessment, model selection and prompt-tuning separation.

A stronger evaluation would use qualified reviewers, authoritative source
material, blinded comparisons, explicit hallucination and completeness labels,
and measured uncertainty. Additional training alone would not resolve flawed
references or missing source context.
