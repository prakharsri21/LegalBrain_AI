"""Prepare deterministic, question-grouped Stage 5 chunks from Stage 4.

The Stage 4 train and validation JSONL files are streamed alongside the Stage 4
manifest. Group selection and its random ranks live in a disk-backed SQLite
index; only the selected 101,000 group metadata rows are held in Python memory.
The test split is never opened or exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("data/processed/stage4-export-v1")
DEFAULT_OUTPUT = Path("data/training_stage5_100k_v1")
SAMPLE_SEED = 42
TRAIN_GROUPS = 100_000
GROUPS_PER_CHUNK = 20_000
VALIDATION_GROUPS = 1_000
CHUNK_COUNT = TRAIN_GROUPS // GROUPS_PER_CHUNK
SUPPORTED_SPLITS = {"train", "validation"}


def rank_for(split: str, seed: int, group_id: str) -> bytes:
    """Return a stable pseudorandom order key, scoped by split and seed."""
    payload = f"legalbrain-stage5-v1:{split}:{seed}:{group_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def validate_chat_record(record: dict[str, Any], path: Path, line_no: int) -> None:
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or [item.get("role") for item in messages] != ["user", "assistant"]
        or not all(isinstance(item.get("content"), str) for item in messages)
    ):
        raise ValueError(f"Invalid Stage 4 messages record at {path}:{line_no}")
    if set(record) != {"messages"} or any(
        set(item) != {"role", "content"} for item in messages
    ):
        raise ValueError(f"Unexpected fields/context at {path}:{line_no}")


def percentile(db: sqlite3.Connection, split: str, p: float, count: int) -> int | None:
    if count == 0:
        return None
    offset = max(0, min(count - 1, int((p * count + 0.999999999) // 1) - 1))
    row = db.execute(
        "SELECT token_length FROM selected_lengths WHERE split=? "
        "ORDER BY token_length LIMIT 1 OFFSET ?",
        (split, offset),
    ).fetchone()
    return int(row[0]) if row else None


def length_statistics(db: sqlite3.Connection, split: str, count: int) -> dict[str, int | None]:
    if count == 0:
        return {key: None for key in ("median", "p90", "p95", "p99", "max")}
    middle = count // 2
    if count % 2:
        median = percentile(db, split, (middle + 1) / count, count)
    else:
        low = db.execute(
            "SELECT token_length FROM selected_lengths WHERE split=? "
            "ORDER BY token_length LIMIT 1 OFFSET ?",
            (split, middle - 1),
        ).fetchone()[0]
        high = db.execute(
            "SELECT token_length FROM selected_lengths WHERE split=? "
            "ORDER BY token_length LIMIT 1 OFFSET ?",
            (split, middle),
        ).fetchone()[0]
        median = statistics.mean((low, high))
    maximum = db.execute(
        "SELECT MAX(token_length) FROM selected_lengths WHERE split=?", (split,)
    ).fetchone()[0]
    return {
        "median": median,
        "p90": percentile(db, split, 0.90, count),
        "p95": percentile(db, split, 0.95, count),
        "p99": percentile(db, split, 0.99, count),
        "max": int(maximum),
    }


def _new_index(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-65536")  # Bound SQLite's page cache to ~64 MiB.
    db.execute("PRAGMA mmap_size=0")
    db.execute(
        """CREATE TABLE groups (
               split TEXT NOT NULL,
               group_id TEXT NOT NULL,
               sample_rank BLOB NOT NULL,
               record_count INTEGER NOT NULL,
               first_manifest_row INTEGER NOT NULL,
               PRIMARY KEY(split, group_id)
           ) WITHOUT ROWID"""
    )
    return db


def prepare(source: Path, output: Path, seed: int = SAMPLE_SEED) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    summary_path = source / "stage4_summary.json"
    manifest_path = source / "manifest.jsonl"
    train_path = source / "train.jsonl"
    valid_path = source / "validation.jsonl"
    for required in (summary_path, manifest_path, train_path, valid_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required Stage 4 input missing: {required}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage 5 output: {output}")

    stage4_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_rows = {
        split: int(stage4_summary["splits"][split]["records"])
        for split in ("train", "validation")
    }
    expected_groups = {
        split: int(stage4_summary["splits"][split]["groups"])
        for split in ("train", "validation")
    }
    if expected_groups["train"] < TRAIN_GROUPS:
        raise ValueError("Stage 4 train split has fewer than 100,000 groups")
    if expected_groups["validation"] < VALIDATION_GROUPS:
        raise ValueError("Stage 4 validation split has fewer than 1,000 groups")

    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    db_path = build_dir / "selection.sqlite3"
    db = _new_index(db_path)
    created_final = False
    try:
        db.execute("BEGIN")
        source_record_rows = Counter()
        with manifest_path.open(encoding="utf-8") as source_manifest:
            for row_no, line in enumerate(source_manifest, 1):
                meta = json.loads(line)
                split = meta.get("split")
                if split not in SUPPORTED_SPLITS:
                    continue
                gid = meta.get("group_id")
                if not isinstance(gid, str) or not gid:
                    raise ValueError(f"Missing Stage 4 group_id at manifest row {row_no}")
                namespace = split
                db.execute(
                    """INSERT INTO groups(split,group_id,sample_rank,record_count,first_manifest_row)
                       VALUES(?,?,?,?,?)
                       ON CONFLICT(split,group_id) DO UPDATE SET
                         record_count=groups.record_count+1""",
                    (split, gid, rank_for(namespace, seed, gid), 1, row_no),
                )
                source_record_rows[split] += 1
                if row_no % 500_000 == 0:
                    db.commit()
                    print(f"Indexed {row_no:,} Stage 4 manifest rows", flush=True)
                    db.execute("BEGIN")
        db.commit()

        actual_groups = {
            split: int(db.execute("SELECT COUNT(*) FROM groups WHERE split=?", (split,)).fetchone()[0])
            for split in SUPPORTED_SPLITS
        }
        for split in SUPPORTED_SPLITS:
            if source_record_rows[split] != expected_rows[split]:
                raise ValueError(
                    f"{split} manifest has {source_record_rows[split]:,} rows; "
                    f"Stage 4 summary says {expected_rows[split]:,}"
                )
            if actual_groups[split] != expected_groups[split]:
                raise ValueError(
                    f"{split} manifest has {actual_groups[split]:,} groups; "
                    f"Stage 4 summary says {expected_groups[split]:,}"
                )
        db.execute("CREATE INDEX groups_sample_order ON groups(split,sample_rank,group_id)")
        db.execute(
            """CREATE TABLE selected_groups (
                 split TEXT NOT NULL, group_id TEXT NOT NULL, sample_rank BLOB NOT NULL,
                 source_record_count INTEGER NOT NULL, sample_position INTEGER NOT NULL,
                 chunk INTEGER, PRIMARY KEY(split,group_id)
               ) WITHOUT ROWID"""
        )
        db.execute(
            "CREATE TABLE selected_lengths(split TEXT NOT NULL,chunk INTEGER,token_length INTEGER NOT NULL)"
        )
        db.execute(
            "CREATE INDEX selected_lengths_sort ON selected_lengths(split,chunk,token_length)"
        )
        db.execute("CREATE TABLE written_ids(source_row_id TEXT PRIMARY KEY,split TEXT NOT NULL,chunk INTEGER)")

        targets = {"train": TRAIN_GROUPS, "validation": VALIDATION_GROUPS}
        for split, target in targets.items():
            cursor = db.execute(
                "SELECT group_id,sample_rank,record_count FROM groups WHERE split=? "
                "ORDER BY sample_rank,group_id LIMIT ?",
                (split, target),
            )
            for position, (gid, sample_rank, record_count) in enumerate(cursor, 1):
                chunk = ((position - 1) // GROUPS_PER_CHUNK + 1) if split == "train" else None
                db.execute(
                    "INSERT INTO selected_groups VALUES(?,?,?,?,?,?)",
                    (split, gid, sample_rank, record_count, position, chunk),
                )
        db.commit()

        # The dictionaries contain only the selected 100,000 + 1,000 groups.
        selected_train: dict[str, tuple[int, int, str, int]] = {}
        selected_valid: dict[str, tuple[int, str, int]] = {}
        for gid, chunk, record_count, sample_rank, position in db.execute(
            "SELECT group_id,chunk,source_record_count,sample_rank,sample_position "
            "FROM selected_groups WHERE split='train'"
        ):
            selected_train[gid] = (int(chunk), int(record_count), bytes(sample_rank).hex(), int(position))
        for gid, record_count, sample_rank, position in db.execute(
            "SELECT group_id,source_record_count,sample_rank,sample_position "
            "FROM selected_groups WHERE split='validation'"
        ):
            selected_valid[gid] = (int(record_count), bytes(sample_rank).hex(), int(position))
        if len(selected_train) != TRAIN_GROUPS or len(selected_valid) != VALIDATION_GROUPS:
            raise AssertionError("Deterministic group selection returned an unexpected count")

        chunks_dir = build_dir / "chunks"
        validation_dir = build_dir / "validation_sample"
        validation_dir.mkdir(parents=True)
        handles: dict[int, Any] = {}
        for chunk in range(1, CHUNK_COUNT + 1):
            chunk_dir = chunks_dir / f"chunk-{chunk:02d}"
            chunk_dir.mkdir(parents=True)
            handles[chunk] = (chunk_dir / "train.jsonl").open("w", encoding="utf-8")
        valid_out = (validation_dir / "valid.jsonl").open("w", encoding="utf-8")
        group_manifest = (build_dir / "group_selection_manifest.jsonl").open("w", encoding="utf-8")
        record_manifest = (build_dir / "record_manifest.jsonl").open("w", encoding="utf-8")
        record_counts = Counter()
        observed_groups = {"train": set(), "validation": set()}
        source_lines_consumed = Counter()
        try:
            with manifest_path.open(encoding="utf-8") as source_manifest, \
                 train_path.open(encoding="utf-8") as train_in, \
                 valid_path.open(encoding="utf-8") as valid_in:
                for row_no, line in enumerate(source_manifest, 1):
                    meta = json.loads(line)
                    split = meta.get("split")
                    if split not in SUPPORTED_SPLITS:
                        # Do not open test.jsonl or use test examples.
                        continue
                    data_file = train_in if split == "train" else valid_in
                    raw_record = data_file.readline()
                    if not raw_record:
                        raise ValueError(f"Unexpected EOF in Stage 4 {split}.jsonl")
                    source_lines_consumed[split] += 1
                    record = json.loads(raw_record)
                    validate_chat_record(record, train_path if split == "train" else valid_path, source_lines_consumed[split])
                    gid = meta["group_id"]
                    if split == "train":
                        selected = selected_train.get(gid)
                        if selected is None:
                            continue
                        chunk, expected_group_records, sample_rank, sample_position = selected
                        handles[chunk].write(raw_record if raw_record.endswith("\n") else raw_record + "\n")
                        observed_groups[split].add(gid)
                        record_counts[f"train_chunk_{chunk}"] += 1
                        record_manifest.write(json.dumps({
                            "source_row_id": meta["source_row_id"],
                            "source_shard": meta["source_shard"],
                            "source_row_number_1_based": meta["source_row_number_1_based"],
                            "group_id": gid,
                            "stage4_split": split,
                            "stage5_chunk": f"chunk-{chunk:02d}",
                            "group_sample_position": sample_position,
                            "group_sample_rank_sha256": sample_rank,
                            "token_length": meta["token_length"],
                        }, separators=(",", ":")) + "\n")
                        db.execute("INSERT INTO written_ids VALUES(?,?,?)", (meta["source_row_id"], split, chunk))
                        db.execute(
                            "INSERT INTO selected_lengths VALUES(?,?,?)",
                            (split, chunk, int(meta["token_length"])),
                        )
                    else:
                        selected = selected_valid.get(gid)
                        if selected is None:
                            continue
                        expected_group_records, sample_rank, sample_position = selected
                        valid_out.write(raw_record if raw_record.endswith("\n") else raw_record + "\n")
                        observed_groups[split].add(gid)
                        record_counts["validation"] += 1
                        record_manifest.write(json.dumps({
                            "source_row_id": meta["source_row_id"],
                            "source_shard": meta["source_shard"],
                            "source_row_number_1_based": meta["source_row_number_1_based"],
                            "group_id": gid,
                            "stage4_split": split,
                            "stage5_chunk": "validation_sample",
                            "group_sample_position": sample_position,
                            "group_sample_rank_sha256": sample_rank,
                            "token_length": meta["token_length"],
                        }, separators=(",", ":")) + "\n")
                        db.execute("INSERT INTO written_ids VALUES(?,?,NULL)", (meta["source_row_id"], split))
                        db.execute(
                            "INSERT INTO selected_lengths VALUES(?,?,?)",
                            (split, None, int(meta["token_length"])),
                        )
                    if sum(record_counts.values()) % 50_000 == 0:
                        db.commit()
                if train_in.readline():
                    raise ValueError("Stage 4 train.jsonl contains records beyond its manifest rows")
                if valid_in.readline():
                    raise ValueError("Stage 4 validation.jsonl contains records beyond its manifest rows")
            db.commit()
        finally:
            for handle in handles.values():
                handle.close()
            valid_out.close()
            group_manifest.close()
            record_manifest.close()

        if source_lines_consumed != source_record_rows:
            raise ValueError(f"Manifest/data line reconciliation failed: {source_lines_consumed} vs {source_record_rows}")
        if observed_groups["train"] != set(selected_train):
            raise ValueError("At least one selected train group was not found during export")
        if observed_groups["validation"] != set(selected_valid):
            raise ValueError("At least one selected validation group was not found during export")

        # Verify that all source records from each selected group were retained.
        expected_train_records_by_chunk = Counter()
        expected_multi_groups_by_chunk = Counter()
        with (build_dir / "group_selection_manifest.jsonl").open("a", encoding="utf-8") as group_manifest:
            for gid, (chunk, group_records, _rank, position) in selected_train.items():
                expected_train_records_by_chunk[chunk] += group_records
                if group_records > 1:
                    expected_multi_groups_by_chunk[chunk] += 1
                group_manifest.write(json.dumps({
                    "stage4_split": "train",
                    "group_id": gid,
                    "chunk": f"chunk-{chunk:02d}",
                    "sample_position": position,
                    "sample_rank_sha256": _rank,
                    "source_record_count": group_records,
                }, separators=(",", ":")) + "\n")
        expected_valid_records = 0
        valid_multi_groups = 0
        with (build_dir / "group_selection_manifest.jsonl").open("a", encoding="utf-8") as group_manifest:
            for gid, (group_records, sample_rank, position) in selected_valid.items():
                expected_valid_records += group_records
                valid_multi_groups += group_records > 1
                group_manifest.write(json.dumps({
                    "stage4_split": "validation",
                    "group_id": gid,
                    "chunk": "validation_sample",
                    "sample_position": position,
                    "sample_rank_sha256": sample_rank,
                    "source_record_count": group_records,
                }, separators=(",", ":")) + "\n")
        for chunk in range(1, CHUNK_COUNT + 1):
            if record_counts[f"train_chunk_{chunk}"] != expected_train_records_by_chunk[chunk]:
                raise ValueError(f"Chunk {chunk} record count does not reconcile to selected group variants")
        if record_counts["validation"] != expected_valid_records:
            raise ValueError("Validation record count does not reconcile to selected group variants")
        duplicate_ids = db.execute(
            "SELECT COUNT(*) FROM (SELECT source_row_id FROM written_ids GROUP BY source_row_id HAVING COUNT(*)>1)"
        ).fetchone()[0]
        if duplicate_ids:
            raise ValueError(f"Selected source IDs are duplicated: {duplicate_ids}")

        # Preserve a fixed validation JSONL beside each MLX-LM chunk. Each is a
        # byte-identical copy of the separate validation_sample/valid.jsonl.
        shared_valid = validation_dir / "valid.jsonl"
        for chunk in range(1, CHUNK_COUNT + 1):
            shutil.copyfile(shared_valid, chunks_dir / f"chunk-{chunk:02d}" / "valid.jsonl")

        for split, target in (("train", "train"), ("validation", "validation")):
            rank_rows = db.execute(
                "SELECT COUNT(*) FROM selected_lengths WHERE split=?", (split,)
            ).fetchone()[0]
            if rank_rows != (sum(expected_train_records_by_chunk.values()) if split == "train" else expected_valid_records):
                raise AssertionError(f"Length metadata does not reconcile for {split}")

        max_sequence_length = 1024
        chunks = {}
        for chunk in range(1, CHUNK_COUNT + 1):
            train_records = int(record_counts[f"train_chunk_{chunk}"])
            chunks[f"chunk-{chunk:02d}"] = {
                "question_groups": GROUPS_PER_CHUNK,
                "records": train_records,
                "multiple_response_groups": int(expected_multi_groups_by_chunk[chunk]),
                "records_over_max_sequence_length": 0,
                "train_file": f"chunks/chunk-{chunk:02d}/train.jsonl",
                "validation_file": f"chunks/chunk-{chunk:02d}/valid.jsonl",
            }
        train_stats = length_statistics(db, "train", sum(expected_train_records_by_chunk.values()))
        validation_stats = length_statistics(db, "validation", expected_valid_records)
        counts_by_chunk_lengths = {}
        for chunk in range(1, CHUNK_COUNT + 1):
            chunk_total = int(db.execute(
                "SELECT COUNT(*) FROM selected_lengths WHERE split='train' AND chunk=?",
                (chunk,),
            ).fetchone()[0])
            chunk_stats = {}
            if chunk_total % 2 == 0 and chunk_total:
                middle = chunk_total // 2
                values = [db.execute(
                    "SELECT token_length FROM selected_lengths WHERE split='train' AND chunk=? "
                    "ORDER BY token_length LIMIT 1 OFFSET ?", (chunk, offset),
                ).fetchone()[0] for offset in (middle - 1, middle)]
                chunk_stats["median"] = statistics.mean(values)
            elif chunk_total:
                middle = chunk_total // 2
                row = db.execute(
                    "SELECT token_length FROM selected_lengths WHERE split='train' AND chunk=? "
                    "ORDER BY token_length LIMIT 1 OFFSET ?", (chunk, middle),
                ).fetchone()
                chunk_stats["median"] = int(row[0])
            else:
                chunk_stats["median"] = None
            if chunk_total:
                p99_offset = max(0, min(chunk_total - 1, int((0.99 * chunk_total + 0.999999999) // 1) - 1))
                row = db.execute(
                    "SELECT token_length FROM selected_lengths WHERE split='train' AND chunk=? "
                    "ORDER BY token_length LIMIT 1 OFFSET ?", (chunk, p99_offset),
                ).fetchone()
                chunk_stats["p99"] = int(row[0])
            else:
                chunk_stats["p99"] = None
            chunk_stats["max"] = int(db.execute(
                "SELECT MAX(token_length) FROM selected_lengths WHERE split='train' AND chunk=?",
                (chunk,),
            ).fetchone()[0]) if chunk_total else None
            chunk_stats["above_1024"] = int(db.execute(
                "SELECT COUNT(*) FROM selected_lengths WHERE split='train' AND chunk=? AND token_length>?",
                (chunk, max_sequence_length),
            ).fetchone()[0])
            counts_by_chunk_lengths[f"chunk-{chunk:02d}"] = chunk_stats
            chunks[f"chunk-{chunk:02d}"]["records_over_max_sequence_length"] = chunk_stats["above_1024"]

        summary = {
            "source": str(source),
            "source_stage4_seed": stage4_summary.get("seed"),
            "source_dataset_revision": stage4_summary.get("dataset_revision"),
            "selection_seed": seed,
            "selection_method": "Deterministic SHA-256 pseudorandom rank by split and normalized-question group_id; lowest ranks selected.",
            "selected_train_question_groups": TRAIN_GROUPS,
            "available_stage4_train_groups": actual_groups["train"],
            "train_chunks": chunks,
            "train_question_groups": sum(item["question_groups"] for item in chunks.values()),
            "train_records": sum(expected_train_records_by_chunk.values()),
            "train_multiple_response_groups": sum(expected_multi_groups_by_chunk.values()),
            "train_token_lengths": train_stats,
            "chunk_token_lengths": counts_by_chunk_lengths,
            "validation": {
                "source_split": "Stage 4 validation only",
                "selection_seed": seed,
                "sampling_namespace": "validation",
                "question_groups": VALIDATION_GROUPS,
                "available_stage4_validation_groups": actual_groups["validation"],
                "records": expected_valid_records,
                "multiple_response_groups": int(valid_multi_groups),
                "token_lengths": validation_stats,
                "file": "validation_sample/valid.jsonl",
                "chunk_local_identical_copies": [f"chunks/chunk-{i:02d}/valid.jsonl" for i in range(1, CHUNK_COUNT + 1)],
            },
            "group_selection_manifest": "group_selection_manifest.jsonl",
            "record_provenance_manifest": "record_manifest.jsonl",
            "max_sequence_length_for_first_run": max_sequence_length,
            "records_over_max_sequence_length_in_selected_training": sum(v["above_1024"] for v in counts_by_chunk_lengths.values()),
            "test_split_used": False,
            "source_stage4_files_modified": False,
            "all_selected_groups_have_all_source_records": True,
            "source_id_uniqueness_check": "passed",
            "manifest_to_source_row_counts_reconciled": True,
            "notes": [
                "Every selected question group and all of its answer variants are assigned to exactly one chunk.",
                "The validation sample is selected only from the independent Stage 4 validation split; no test JSONL was opened.",
                "MLX-LM's local loader reads the current chunk's train and valid JSONL files into Python lists. Tokenization is cached lazily for examples seen during training; only one 20k-group chunk is loaded per run.",
                "MLX-LM's batch iterator permutes batches each epoch. The chunk membership itself was selected by deterministic pseudorandom hash ranks.",
                "The initial run should use base weights because the blind AI review did not show a clear advantage for the pilot adapter.",
                "MLX-LM adapter resume loads adapter weights only; optimizer state and global step are not restored by --resume-adapter-file.",
                "Training with max_seq_length=1024 truncates the small number of selected sequences exceeding 1024 tokens; raw Stage 4 source remains intact and over-limit counts are reported.",
            ],
        }
        (build_dir / "stage5_selection_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (build_dir / "stage5_preparation_report.md").write_text(_report(summary), encoding="utf-8")
        db.close()
        db_path.unlink(missing_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(build_dir, output)
        created_final = True
        return summary
    finally:
        try:
            db.close()
        except Exception:
            pass
        if not created_final and build_dir.exists():
            shutil.rmtree(build_dir)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 5 100k-group preparation",
        "",
        f"Stage 4 source: `{summary['source']}`",
        f"Dataset revision: `{summary['source_dataset_revision']}`",
        f"Group sampling seed: `{summary['selection_seed']}`",
        "",
        "Training question groups were selected from Stage 4 train only using stable SHA-256 ranks. All response variants for a group remain together. Each 20,000-group chunk is a different segment of the same seeded pseudorandom ranking. MLX-LM permutes training batches each epoch.",
        "",
        "| Chunk | Groups | Records | Multiple-response groups | Token median | P99 | Max | >1,024 tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, chunk in summary["train_chunks"].items():
        stats = summary["chunk_token_lengths"][name]
        lines.append(
            f"| {name} | {chunk['question_groups']:,} | {chunk['records']:,} | "
            f"{chunk['multiple_response_groups']:,} | {stats['median']} | {stats['p99']} | "
            f"{stats['max']} | {stats['above_1024']:,} |"
        )
    val = summary["validation"]
    lines += [
        "",
        f"The fixed validation sample contains {val['question_groups']:,} groups and {val['records']:,} records, selected only from Stage 4 validation (seed {val['selection_seed']}). The same validation JSONL is copied byte-for-byte into each chunk's `valid.jsonl` for MLX-LM compatibility.",
        "",
        f"Stage 4 train groups available: {summary['available_stage4_train_groups']:,}; selected: {summary['selected_train_question_groups']:,}. Total selected training records: {summary['train_records']:,}.",
        "",
        "No Stage 4 test examples or test JSONL were used. Source Stage 4 files were read-only. See `record_manifest.jsonl` for source IDs/shards/row numbers and `group_selection_manifest.jsonl` for group-to-chunk assignment and per-group record counts.",
        "",
        "## MLX-LM memory and recovery notes",
        "",
        "MLX-LM 0.32.2's local loader reads each selected `train.jsonl`/`valid.jsonl` into Python lists; tokenization is lazy and `CacheDataset` retains processed examples during a run. Running one 20k-group chunk at a time keeps those lists and token caches limited to the active chunk. The loader is not streaming across the full 100k set.",
        "",
        "The batch iterator sorts examples by the dataset's item-length function and randomly permutes batches each epoch. For the chat dataset wrapper this sort key is the JSON record field count, so batches are effectively randomized from the input order. The preparation rank shuffles group membership across chunks reproducibly.",
        "",
        "`--resume-adapter-file` loads adapter weights only. MLX-LM creates a new Adam optimizer for each CLI process; it does not restore optimizer moments or global iteration state. Checkpoints are recoverable adapter weights, not complete optimizer-state snapshots.",
        "",
        "The first run is planned from the original Qwen3-4B base model. The blind AI-scored pilot showed no meaningful advantage for the previous adapter. `max_seq_length=1024` follows the Stage 4 recommendation; selected examples longer than 1,024 tokens are counted and MLX-LM truncates them in the training batch, without modifying source files.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
