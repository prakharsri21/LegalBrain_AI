# Project structure

[Overview](../README.md) · [Pipeline](DATA_PIPELINE.md) · [Reproducibility](REPRODUCIBILITY.md)

```text
app/                         FastAPI backend and Gradio interface
  services/                  Request guardrails and bounded response cache
scripts/data_pipeline/
  audit/                     Corpus audit, language validation, answer-group review
  clean/                     Cleaning policy, language rules and Stage 3 runner
  export/                    Stage 4 group splitting and JSONL export
scripts/                     Training preparation, evaluation and older utilities
tests/                       Unit tests for pipeline and app services
docs/                        Pipeline, experiments and reproducibility
README.md                    Project overview
pyproject.toml, uv.lock      Dependencies and resolved versions
```

`data/` and `outputs/` are local and ignored: raw shards, processed partitions,
reports, experiment samples, model adapters and logs are not bundled in Git.
Copies of these artifacts require independent storage and restoration. The legacy
scripts remain available; the Stage 3 and Stage 4 implementations are in the
pipeline subdirectories. Empty local placeholders have no runtime role.
