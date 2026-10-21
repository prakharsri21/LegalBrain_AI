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
FIELDS = ("context", "question", "response")
SHORT_ANSWER_LIMIT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small, reproducible dataset for MLX-LM fine-tuning."
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
        help="Maximum sampled records from each shard (default: 200).",
    )
    parser.add_argument(
        "--scan-multiplier",
        type=int,
        default=3,
        help="Records scanned per shard relative to the requested sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory (default: data/processed).",
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
        parser.error("--shards must be comma-separated integers")

    if not args.shards:
        parser.error("Provide at least one shard index")
    if len(set(args.shards)) != len(args.shards):
        parser.error("--shards must not contain duplicate indices")
    if any(index < 0 or index >= SHARD_COUNT for index in args.shards):
        parser.error(f"Shard indices must be between 0 and {SHARD_COUNT - 1}")

    return args


def normalize_whitespace(value: str) -> str:
    """Normalize whitespace for comparisons without changing meaning."""
    return re.sub(r"\s+", " ", value).strip()


def normalize_context(value: str) -> str:
    """Normalize spaces while preserving paragraph and line breaks."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def sample_shard(
    shard_index: int,
    rows_to_sample: int,
    scan_multiplier: int,
    seed: int,
) -> tuple[list[dict], int]:
    """
    Sample from a bounded window of one shard.

    Reservoir sampling prevents keeping every scanned record in memory.
    The result represents only the scanned window of this shard, not
    the full dataset.
    """
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
            f"Failed to read shard {shard_index}. "
            "Check network access and dataset availability.\n"
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
    """
    Remove examples with blank required fields and exact duplicate triples.

    Short answers are retained and reported separately for manual review.
    """
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    blank_removed = 0
    duplicate_removed = 0

    for row in rows:
        record = {
            "context": normalize_context(
                str(row.get("context") or "")
            ),
            "question": normalize_whitespace(
                str(row.get("question") or "")
            ),
            "response": normalize_whitespace(
                str(row.get("response") or "")
            ),
        }

        if not all(record[field] for field in FIELDS):
            blank_removed += 1
            continue

        # Deduplicate on all three fields after whitespace normalization.
        duplicate_key = tuple(
            record[field].casefold() for field in FIELDS
        )

        if duplicate_key in seen:
            duplicate_removed += 1
            continue

        seen.add(duplicate_key)
        cleaned.append(record)

    return cleaned, blank_removed, duplicate_removed


def make_chat_example(record: dict[str, str]) -> dict:
    """
    Convert a legal QA triple to the messages format used by MLX-LM.

    The user message supplies the source context and the question.
    The assistant message contains the reference answer.
    """
    user_message = (
        "Answer the legal question using only the supplied context. "
        "If the context does not provide enough information, say so "
        "rather than inventing facts.\n\n"
        f"Legal context:\n{record['context']}\n\n"
        f"Question:\n{record['question']}"
    )

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": record["response"]},
        ]
    }


def grouped_split(
    records: list[dict[str, str]],
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """
    Split by normalized context to reduce leakage between splits.

    Examples sharing the same context stay in the same split. Ratios are
    approximate because context groups may contain different numbers
    of examples.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)

    for record in records:
        context_key = normalize_whitespace(
            record["context"]
        ).casefold()
        groups[context_key].append(record)

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

    # Put each entire context group into the split most below its target.
    for group in group_list:
        selected_split = max(
            splits,
            key=lambda name: targets[name] - len(splits[name]),
        )
        splits[selected_split].extend(group)

    return splits


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    """Write one serialized chat example per line."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(
                make_chat_example(record),
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def main() -> None:
    args = parse_args()

    print(f"Dataset: {DATASET_ID}")
    print(f"Shards: {args.shards}")
    print(f"Rows requested per shard: {args.rows_per_shard}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("\nStreaming enabled; the full corpus will not be downloaded.")

    sampled_rows: list[dict] = []
    scan_report: dict[str, dict[str, int]] = {}

    # Read one shard at a time to avoid the cross-shard shuffle behavior
    # that caused problems during our earlier audit.
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

    print(f"\nTotal records sampled: {len(sampled_rows)}")

    records, blank_removed, duplicate_removed = clean_records(
        sampled_rows
    )

    if len(records) < 10:
        raise SystemExit(
            f"Only {len(records)} usable records remain. "
            "Inspect the data before creating training splits."
        )

    short_records = [
        record
        for record in records
        if len(record["response"]) < SHORT_ANSWER_LIMIT
    ]

    print("\n--- Cleaning summary ---")
    print(f"Blank-field records removed: {blank_removed}")
    print(f"Exact duplicate records removed: {duplicate_removed}")
    print(f"Records retained: {len(records)}")
    print(
        f"Responses under {SHORT_ANSWER_LIMIT} characters: "
        f"{len(short_records)}"
    )

    if short_records:
        print("\nShort-answer examples for manual review:")
        for record in short_records[:5]:
            print(f"- {record['response'][:180]}")

    splits = grouped_split(records, seed=args.seed)

    # Shuffle examples within each split reproducibly.
    rng = random.Random(args.seed)
    for examples in splits.values():
        rng.shuffle(examples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "valid", "test"):
        path = output_dir / f"{split_name}.jsonl"
        write_jsonl(path, splits[split_name])
        print(f"{split_name}: {len(splits[split_name])} examples -> {path}")

    # Save metadata, not the raw dataset, alongside the prepared files.
    manifest = {
        "dataset_id": DATASET_ID,
        "shards": args.shards,
        "scan_report": scan_report,
        "seed": args.seed,
        "requested_rows_per_shard": args.rows_per_shard,
        "scan_multiplier": args.scan_multiplier,
        "sampled_records": len(sampled_rows),
        "blank_records_removed": blank_removed,
        "exact_duplicates_removed": duplicate_removed,
        "retained_records": len(records),
        "short_responses_retained": len(short_records),
        "grouping_method": "normalized context",
        "split_counts": {
            name: len(examples)
            for name, examples in splits.items()
        },
        "split_ratios_target": {
            "train": 0.80,
            "valid": 0.10,
            "test": 0.10,
        },
        "limitations": [
            "Samples come from bounded windows of selected shards.",
            "The sample is not representative of the full corpus.",
            "Short answers are flagged, not automatically removed.",
            "Exact-duplicate removal does not detect paraphrases.",
            "Split ratios are approximate due to context grouping.",
        ],
    }

    manifest_path = output_dir / "summary.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"\nSummary written to {manifest_path}")
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()