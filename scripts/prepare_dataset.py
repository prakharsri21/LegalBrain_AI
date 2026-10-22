from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "Prarabdha/indian-legal-supervised-fine-tuning-data"
SHARD_COUNT = 33
DEFAULT_SHARDS = "0,10,20"
SHORT_ANSWER_LIMIT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare question-only legal QA data for MLX-LM."
    )
    parser.add_argument(
        "--shards",
        default=DEFAULT_SHARDS,
        help="Comma-separated shard indices (default: 0,10,20).",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=200,
        help="Maximum sampled rows per shard.",
    )
    parser.add_argument(
        "--scan-multiplier",
        type=int,
        default=3,
        help="Maximum records scanned per requested sample row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and splitting.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_question_only",
        help="Directory for the new question-only dataset.",
    )

    args = parser.parse_args()

    if args.rows_per_shard < 1:
        parser.error("--rows-per-shard must be greater than zero")

    if args.scan_multiplier < 1:
        parser.error("--scan-multiplier must be greater than zero")

    try:
        args.shards = [
            int(part.strip())
            for part in args.shards.split(",")
            if part.strip()
        ]
    except ValueError:
        parser.error("--shards must contain comma-separated integers")

    if not args.shards:
        parser.error("Provide at least one shard index")

    if len(set(args.shards)) != len(args.shards):
        parser.error("--shards must not contain duplicate indices")

    if any(index < 0 or index >= SHARD_COUNT for index in args.shards):
        parser.error(f"Shard indices must be between 0 and {SHARD_COUNT - 1}")

    return args


def normalize_text(text: str) -> str:
    """Normalize whitespace for comparisons."""
    return re.sub(r"\s+", " ", text).strip()


def sample_shard(
    shard_index: int,
    rows_to_sample: int,
    scan_multiplier: int,
    seed: int,
) -> tuple[list[dict], int]:
    """Reservoir-sample from a bounded window of one remote shard."""
    shard_path = (
        f"data/train-{shard_index:05d}-of-{SHARD_COUNT:05d}.parquet"
    )

    scan_limit = rows_to_sample * scan_multiplier
    rng = random.Random(seed + shard_index)
    reservoir: list[dict] = []
    scanned = 0
    iterator = None

    print(f"\nReading shard {shard_index}: {shard_path}")
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

            if len(reservoir) < rows_to_sample:
                reservoir.append(row)
            else:
                replacement_index = rng.randrange(scanned + 1)
                if replacement_index < rows_to_sample:
                    reservoir[replacement_index] = row

            scanned += 1

    except Exception as exc:
        raise RuntimeError(
            f"Failed to read shard {shard_index}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    finally:
        if iterator is not None:
            close_iterator = getattr(iterator, "close", None)
            if callable(close_iterator):
                with suppress(Exception):
                    close_iterator()

    print(f"Records scanned: {scanned}")
    print(f"Records sampled: {len(reservoir)}")

    return reservoir, scanned


def clean_records(
    rows: list[dict],
) -> tuple[list[dict[str, str]], int, int]:
    """Remove incomplete and exact-duplicate source examples."""
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    blank_removed = 0
    duplicate_removed = 0

    for row in rows:
        context = normalize_text(str(row.get("context") or ""))
        question = normalize_text(str(row.get("question") or ""))
        response = normalize_text(str(row.get("response") or ""))

        if not context or not question or not response:
            blank_removed += 1
            continue

        # Keep the source context internally for split grouping.
        # It will not be placed in the training prompt.
        record = {
            "context": context,
            "question": question,
            "response": response,
        }

        duplicate_key = (
            context.casefold(),
            question.casefold(),
            response.casefold(),
        )

        if duplicate_key in seen:
            duplicate_removed += 1
            continue

        seen.add(duplicate_key)
        cleaned.append(record)

    return cleaned, blank_removed, duplicate_removed


def grouped_split(
    records: list[dict[str, str]],
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """
    Split related records together by normalized source context.

    This reduces leakage between splits when multiple Q&A pairs came
    from the same source passage. Split percentages are approximate.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for record in records:
        key = normalize_text(record["context"]).casefold()
        groups[key].append(record)

    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)

    total = len(records)
    targets = {
        "train": total * 0.80,
        "valid": total * 0.10,
        "test": total * 0.10,
    }

    splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for group in group_list:
        selected_split = max(
            splits,
            key=lambda name: targets[name] - len(splits[name]),
        )
        splits[selected_split].extend(group)

    # Prevent the same normalized question from appearing in different
    # splits, even if different source contexts produced it.
    question_owner: dict[str, str] = {}
    clean_splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }

    # Train gets first priority; validation and test remain disjoint
    # from questions already assigned to a split.
    for split_name in ("train", "valid", "test"):
        for record in splits[split_name]:
            question_key = record["question"].casefold()
            owner = question_owner.get(question_key)

            if owner is None:
                question_owner[question_key] = split_name
                clean_splits[split_name].append(record)
            else:
                # Duplicate questions from another split are omitted
                # to avoid direct question leakage.
                continue

    return clean_splits


def make_chat_example(record: dict[str, str]) -> dict:
    """
    Produce question-only supervised fine-tuning data.

    The legal context is deliberately excluded from the user prompt.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": record["question"],
            },
            {
                "role": "assistant",
                "content": record["response"],
            },
        ]
    }


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    make_chat_example(record),
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()

    print(f"Dataset: {DATASET_ID}")
    print(f"Shards: {args.shards}")
    print(f"Rows requested per shard: {args.rows_per_shard}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print("Training format: question → answer (no context in prompts)")
    print("Streaming enabled; the full dataset will not be downloaded.")

    sampled_rows: list[dict] = []
    scan_report: dict[str, dict[str, int]] = {}

    for shard_index in args.shards:
        rows, scanned = sample_shard(
            shard_index=shard_index,
            rows_to_sample=args.rows_per_shard,
            scan_multiplier=args.scan_multiplier,
            seed=args.seed,
        )

        sampled_rows.extend(rows)
        scan_report[str(shard_index)] = {
            "scanned": scanned,
            "sampled": len(rows),
        }

    if not sampled_rows:
        raise SystemExit("No records were sampled.")

    records, blank_removed, duplicate_removed = clean_records(
        sampled_rows
    )

    if len(records) < 10:
        raise SystemExit(
            f"Only {len(records)} valid records remain. "
            "Review the dataset before proceeding."
        )

    short_records = [
        record
        for record in records
        if len(record["response"]) < SHORT_ANSWER_LIMIT
    ]

    print("\n--- Cleaning summary ---")
    print(f"Records sampled: {len(sampled_rows)}")
    print(f"Blank records removed: {blank_removed}")
    print(f"Exact duplicate triples removed: {duplicate_removed}")
    print(f"Valid records retained: {len(records)}")
    print(
        f"Responses shorter than {SHORT_ANSWER_LIMIT} characters: "
        f"{len(short_records)} (retained for now)"
    )

    splits = grouped_split(records, seed=args.seed)

    rng = random.Random(args.seed)
    for examples in splits.values():
        rng.shuffle(examples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "valid", "test"):
        path = output_dir / f"{split_name}.jsonl"
        write_jsonl(path, splits[split_name])
        print(f"{split_name}: {len(splits[split_name])} examples -> {path}")

    manifest = {
        "dataset_id": DATASET_ID,
        "shards": args.shards,
        "scan_report": scan_report,
        "seed": args.seed,
        "requested_rows_per_shard": args.rows_per_shard,
        "scan_multiplier": args.scan_multiplier,
        "sampled_records": len(sampled_rows),
        "blank_records_removed": blank_removed,
        "exact_duplicate_triples_removed": duplicate_removed,
        "retained_source_records": len(records),
        "short_responses_retained": len(short_records),
        "training_input_format": "question_only",
        "source_context_used_for_grouped_splits": True,
        "source_context_in_model_messages": False,
        "split_counts": {
            name: len(examples)
            for name, examples in splits.items()
        },
        "limitations": [
            "Only bounded windows of selected shards were sampled.",
            "The sample is not representative of the entire corpus.",
            "Answers were originally paired with source contexts, "
            "but those contexts are omitted from model prompts.",
            "Short answers are retained and require quality review.",
            "Split ratios are approximate.",
        ],
    }

    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()