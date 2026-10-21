from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from contextlib import suppress
from statistics import mean, median

from datasets import load_dataset


DATASET_ID = "Prarabdha/indian-legal-supervised-fine-tuning-data"
SHARD_COUNT = 33

FIELDS = ("context", "question", "response")

# Heuristic flags only. These do not determine whether a record is valid.
TECH_TERMS = re.compile(
    r"\b(domain name system|DNS|IP address|website|"
    r"programming|database|operating system|computer network)\b",
    re.IGNORECASE,
)

LEGAL_TERMS = re.compile(
    r"\b(act|section|article|court|judgment|petitioner|"
    r"respondent|constitution|statute|supreme court|high court)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a sample from one Indian legal dataset shard."
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of records to sample (default: 100).",
    )

    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help=f"Shard index, from 0 to {SHARD_COUNT - 1} (default: 0).",
    )

    parser.add_argument(
        "--scan-multiplier",
        type=int,
        default=2,
        help=(
            "Number of records to inspect per requested sample row. "
            "A value of 2 samples from a window up to twice the requested size."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible reservoir sampling (default: 42).",
    )

    args = parser.parse_args()

    if args.rows < 1:
        parser.error("--rows must be greater than zero")

    if not 0 <= args.shard_index < SHARD_COUNT:
        parser.error(
            f"--shard-index must be between 0 and {SHARD_COUNT - 1}"
        )

    if args.scan_multiplier < 1:
        parser.error("--scan-multiplier must be at least 1")

    return args


def load_sample(
    rows_to_sample: int,
    shard_index: int,
    scan_multiplier: int,
    seed: int,
) -> tuple[list[dict], int]:
    """
    Read a limited window from one remote Parquet shard.

    Reservoir sampling selects a reproducible sample within the scanned
    window without keeping every scanned record in memory.

    This is NOT a random sample of the complete six-million-row corpus.
    """
    shard_path = (
        f"data/train-{shard_index:05d}-of-{SHARD_COUNT:05d}.parquet"
    )
    scan_limit = rows_to_sample * scan_multiplier
    rng = random.Random(seed)
    sample: list[dict] = []
    scanned = 0
    iterator = None

    print(f"Reading one shard: {shard_path}")
    print(f"Maximum records to scan: {scan_limit}")

    try:
        dataset = load_dataset(
            DATASET_ID,
            data_files={"train": [shard_path]},
            split="train",
            streaming=True,
        )

        iterator = iter(dataset)

        while scanned < scan_limit:
            try:
                row = next(iterator)
            except StopIteration:
                break

            # Standard reservoir sampling: each row in the scanned window
            # has an equal chance of being included in the final sample.
            if len(sample) < rows_to_sample:
                sample.append(row)
            else:
                replacement_index = rng.randrange(scanned + 1)

                if replacement_index < rows_to_sample:
                    sample[replacement_index] = row

            scanned += 1

    except Exception as exc:
        raise RuntimeError(
            "Failed while opening or reading the selected dataset shard. "
            "Check network access and the remote file configuration.\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    finally:
        # Close the iterator where supported.
        if iterator is not None:
            close_iterator = getattr(iterator, "close", None)

            if callable(close_iterator):
                with suppress(Exception):
                    close_iterator()

    return sample, scanned


def normalize_records(rows: list[dict]) -> list[dict[str, str]]:
    return [
        {
            field: str(row.get(field) or "").strip()
            for field in FIELDS
        }
        for row in rows
    ]


def report_completeness(records: list[dict[str, str]]) -> None:
    total = len(records)

    print("\n--- Required-field completeness ---")

    for field in FIELDS:
        missing = sum(not record[field] for record in records)

        print(
            f"{field}: {missing}/{total} blank "
            f"({missing / total * 100:.1f}%)"
        )


def report_lengths(records: list[dict[str, str]]) -> None:
    print("\n--- Character lengths ---")

    for field in FIELDS:
        lengths = [
            len(record[field])
            for record in records
            if record[field]
        ]

        if not lengths:
            print(f"{field}: no non-empty values")
            continue

        print(
            f"{field}: mean={mean(lengths):.0f}, "
            f"median={median(lengths):.0f}, "
            f"min={min(lengths)}, "
            f"max={max(lengths)}"
        )

    total = len(records)
    context_lengths = [len(record["context"]) for record in records]

    near_cap = sum(length >= 2450 for length in context_lengths)
    exact_cap = sum(length == 2500 for length in context_lengths)

    print("\n--- Possible context-length cap ---")
    print(
        f"Contexts >= 2,450 chars: {near_cap}/{total} "
        f"({near_cap / total * 100:.1f}%)"
    )
    print(
        f"Contexts exactly 2,500 chars: {exact_cap}/{total} "
        f"({exact_cap / total * 100:.1f}%)"
    )


def report_duplicates(records: list[dict[str, str]]) -> None:
    questions = [
        re.sub(r"\s+", " ", record["question"]).casefold()
        for record in records
        if record["question"]
    ]

    counts = Counter(questions)
    repeated_unique = sum(count > 1 for count in counts.values())
    duplicate_excess = sum(
        count - 1 for count in counts.values() if count > 1
    )

    print("\n--- Repeated questions ---")
    print(f"Unique questions repeated: {repeated_unique}")
    print(f"Repeated-question excess: {duplicate_excess}")


def report_examples(
    records: list[dict[str, str]],
    limit: int = 3,
) -> None:
    print("\n--- Example previews ---")

    field_limits = {
        "context": 350,
        "question": 250,
        "response": 350,
    }

    for index, record in enumerate(records[:limit], start=1):
        print(f"\nExample {index}")

        for field, max_chars in field_limits.items():
            value = record[field]
            print(f"{field}: {value[:max_chars]}")

            if len(value) > max_chars:
                print("  [...truncated for display]")


def report_domain_flags(
    records: list[dict[str, str]],
    limit: int = 10,
) -> None:
    flagged = []

    for index, record in enumerate(records, start=1):
        contains_tech_term = bool(TECH_TERMS.search(record["question"]))
        combined_text = record["context"] + " " + record["question"]
        contains_legal_term = bool(LEGAL_TERMS.search(combined_text))

        if contains_tech_term and not contains_legal_term:
            flagged.append((index, record))

    print("\n--- Potential out-of-domain examples ---")
    print("These are heuristic flags for manual review only.")
    print(f"Flagged examples: {len(flagged)}")

    for index, record in flagged[:limit]:
        print(f"\nRow {index}")
        print(f"Question: {record['question'][:250]}")
        print(f"Context: {record['context'][:250]}")

    if not flagged:
        print("No examples matched this heuristic.")


def main() -> None:
    args = parse_args()

    print(f"Dataset: {DATASET_ID}")
    print(f"Selected shard index: {args.shard_index}")
    print(f"Requested sample: {args.rows}")
    print(f"Random seed: {args.seed}")
    print("Streaming enabled; no dataset output files will be created.\n")

    try:
        rows, scanned = load_sample(
            rows_to_sample=args.rows,
            shard_index=args.shard_index,
            scan_multiplier=args.scan_multiplier,
            seed=args.seed,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not rows:
        raise SystemExit("The selected shard returned no records.")

    records = normalize_records(rows)

    print(f"\nRecords scanned: {scanned}")
    print(f"Records sampled: {len(records)}")

    report_completeness(records)
    report_lengths(records)
    report_duplicates(records)
    report_examples(records)
    report_domain_flags(records)

    print("\nAudit complete.")


if __name__ == "__main__":
    main()