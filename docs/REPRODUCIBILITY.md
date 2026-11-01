# Reproducibility and local setup

[Overview](../README.md) · [Pipeline](DATA_PIPELINE.md) · [Experiments](TRAINING_AND_EVALUATION.md)

## Environment

Use compatible Apple Silicon, Python 3.13, `uv`, and sufficient local storage.
Run `uv sync` in the repository to install the locked dependencies; the initial
installation and uncached Qwen model require network access. For an already
provisioned environment, checks can run offline:

```bash
uv run --offline python -m unittest discover -s tests -q
uv run --offline python -m compileall -q app scripts tests
```

`mlx-lm` requires an accessible Metal device even for some imports. A sandbox or
headless environment can fail before model loading. Data, reports, model weights
and adapters are Git-ignored; a fresh clone does not contain the local artifacts
or guarantee access to the model cache. Obtain the pinned dataset and model
separately, subject to upstream terms.

## Application

The app defaults to an older local adapter path; explicitly choose `base` if it
is absent, or point to a compatible existing adapter. `LEGALBRAIN_MODEL` may be
a model ID or local snapshot path. Environment settings are read at import/startup.

```bash
LEGALBRAIN_ADAPTER_PATH=base uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
# Separate terminal:
uv run python app/ui.py
```

For the completed adapter, set `LEGALBRAIN_ADAPTER_PATH` to
`outputs/legalbrain-qwen3-4b-stage5-100k-v1-base-chunk01` only after restoring it.
The UI defaults to `http://127.0.0.1:8000`; `LEGALBRAIN_API_URL` overrides it.

## Pipeline entry points

The raw input snapshot is `Prarabdha/indian-legal-supervised-fine-tuning-data`,
revision `fe58e87d8393ddd0b8e4224e3eec086c37f1f78e`. See the [source card](https://huggingface.co/datasets/Prarabdha/indian-legal-supervised-fine-tuning-data)
and [cleaning policy](../scripts/data_pipeline/clean/CLEANING_POLICY.md). The
runners create substantial local outputs; inspect options and choose new output
directories before running. These help commands do not process the corpus:

```bash
uv run --offline python scripts/data_pipeline/audit/run_audit.py --help
uv run --offline python scripts/data_pipeline/clean/run_language_clean.py --help
uv run --offline python scripts/data_pipeline/export/run_stage4_export.py --help
uv run --offline python scripts/prepare_stage5_chunks.py --help
uv run --offline python scripts/evaluate_model.py --help
```

Stage 3 full scans require an explicitly approved policy configuration. Stage 4
needs Stage 3 PASS partitions, the multiple-response audit and the Qwen tokenizer.
The 100k preparation script uses Stage 4 train and validation only. Existing
outputs must be preserved; use a new output directory. The evaluator loads its
input JSONL into memory, so prepare a small sample before evaluating a large
split. Its `--output` refuses to overwrite a file unless explicitly instructed.

The recorded Stage 4 group split seed was `20261031`; Stage 5 selection, training
and the final 30-group test sample used seed `42`. The final test sample is no
longer unseen. A fresh independent holdout is needed for later final evaluation.
