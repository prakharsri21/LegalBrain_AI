# LegalBrain AI — Data Pipeline

The offline data pipeline has three stages.

1. Audit
   Measure data quality and understand the dataset before making
   cleaning decisions. Raw files remain unchanged.

2. Clean
   Apply approved data-quality rules and record accepted,
   rejected, and quarantined records.

3. Export
   Create validated question-only conversational JSONL files
   and leakage-resistant train, validation, and test splits.

Data flow:

Raw Parquet
    |
    v
Audit and Quality Reports
    |
    v
Cleaning and Deduplication
    |
    v
Accepted Intermediate Records
    |
    v
Grouped Splitting and Token Validation
    |
    v
MLX-LM JSONL Files

The pipeline runs offline. It is separate from the FastAPI
application and does not run during chatbot inference.