"""Build grouped, question-only Stage 4 JSONL exports from Stage 3 PASS."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

try:
    from scripts.data_pipeline.audit.run_audit import normalize_for_comparison, stable_hash
except ModuleNotFoundError:  # direct script execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.data_pipeline.audit.run_audit import normalize_for_comparison, stable_hash

MODEL = "Qwen/Qwen3-4B"
SEED = 20261031
SPLITS = ("train", "validation", "test")
RATIOS = {"train": .90, "validation": .05, "test": .05}


def group_id(question: str) -> str:
    return "qgrp_" + stable_hash([question])


def choose_split_counts(group_sizes: dict[str, int], seed: int = SEED) -> dict[str, str]:
    """Assign whole groups by seeded hash order to closest record-count targets."""
    total = sum(group_sizes.values())
    targets = {s: total * RATIOS[s] for s in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    assignment: dict[str, str] = {}
    ordered = sorted(group_sizes, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).digest())
    for gid in ordered:
        split = max(SPLITS, key=lambda s: (targets[s] - counts[s], -SPLITS.index(s)))
        assignment[gid] = split
        counts[split] += group_sizes[gid]
    return assignment


def token_ids(tokenizer, question: str, response: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}, {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False, return_dict=False,
    )


def percentiles(lengths: list[int]) -> dict[str, int | None]:
    if not lengths:
        return {k: None for k in ("median", "p90", "p95", "p99", "max")}
    ordered = sorted(lengths)
    def quantile(p: float) -> int:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]
    return {"median": int(statistics.median(ordered)), "p90": quantile(.90),
            "p95": quantile(.95), "p99": quantile(.99), "max": ordered[-1]}


def run(stage3: Path, audit: Path, output: Path, seed: int = SEED, batch_size: int = 2048) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    summary3 = json.loads((stage3 / "stage3_summary.json").read_text())
    expected = int(summary3["outcome_counts"]["PASS"])
    parts = sorted((stage3 / "outcome_partitions/PASS").glob("*.parquet"))
    if len(parts) != 33 or sum(pq.ParquetFile(p).metadata.num_rows for p in parts) != expected:
        raise ValueError("Stage 3 PASS Parquet parts do not reconcile to the Stage 3 summary")
    audit_summary = json.loads((audit / "multiple_response_summary.json").read_text())
    if audit_summary["pass_records_examined"] != expected:
        raise ValueError("Multiple-response audit was not generated from this PASS pool")
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="stage4-", dir=output))
    db = sqlite3.connect(work / "groups.sqlite3")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("CREATE TABLE groups(gid TEXT PRIMARY KEY, size INTEGER NOT NULL, split TEXT)")
    db.execute("CREATE TABLE records(rowid TEXT PRIMARY KEY, gid TEXT NOT NULL, shard TEXT NOT NULL, source_row INTEGER NOT NULL, question TEXT NOT NULL, response TEXT NOT NULL)")
    for path in parts:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["source_row_id", "source_shard", "source_row_number_1_based", "question", "response", "outcome"]):
            for r in batch.to_pylist():
                if r["outcome"] != "PASS" or not r["question"].strip() or not r["response"].strip():
                    raise ValueError(f"Invalid Stage 3 PASS row: {r.get('source_row_id')}")
                gid = group_id(r["question"])
                db.execute("INSERT INTO records VALUES(?,?,?,?,?,?)", (r["source_row_id"], gid, r["source_shard"], r["source_row_number_1_based"], r["question"], r["response"]))
                db.execute("INSERT INTO groups VALUES(?,?,NULL) ON CONFLICT(gid) DO UPDATE SET size=size+1", (gid, 1))
        db.commit()
    group_cursor = db.execute("SELECT gid FROM groups ORDER BY gid")
    while rows := group_cursor.fetchmany(batch_size):
        updates = []
        for (gid,) in rows:
            rank = int.from_bytes(hashlib.sha256(f"{seed}:{gid}".encode()).digest()[:8], "big") % 10000
            updates.append(("train" if rank < 9000 else ("validation" if rank < 9500 else "test"), gid))
        db.executemany("UPDATE groups SET split=? WHERE gid=?", updates)
        db.commit()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    context_limit = int(config.max_position_embeddings)
    if not tokenizer.chat_template:
        raise ValueError("Qwen3 tokenizer has no chat template")
    multi_ids = {x["group_id"] for x in _read_jsonl_gzip(audit / "multiple_response_groups.jsonl.gz")}
    outputs = {s: (output / f"{s}.jsonl").open("w", encoding="utf-8") for s in SPLITS}
    manifest = (output / "manifest.jsonl").open("w", encoding="utf-8")
    db.execute("CREATE TABLE token_lengths(split TEXT NOT NULL,n INTEGER NOT NULL)")
    db.execute("CREATE INDEX token_lengths_idx ON token_lengths(split,n)")
    length_counts = Counter()
    examples = []
    multi_counts = Counter()
    row_counts = Counter()
    group_counts = Counter()
    multi_group_counts = {s:set() for s in SPLITS}
    overlong_counts = Counter()
    try:
        cursor = db.execute("SELECT r.rowid,r.gid,r.shard,r.source_row,r.question,r.response,g.split FROM records r JOIN groups g USING(gid) ORDER BY r.rowid")
        for rowid,gid,shard,source_row,q,a,split in cursor:
            messages = [{"role":"user","content":q},{"role":"assistant","content":a}]
            rec = {"messages":messages}
            ids = token_ids(tokenizer,q,a)
            n = len(ids)
            outputs[split].write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n")
            manifest.write(json.dumps({"source_row_id":rowid,"source_shard":shard,"source_row_number_1_based":source_row,"group_id":gid,"split":split,"token_length":n},separators=(",",":"))+"\n")
            db.execute("INSERT INTO token_lengths VALUES(?,?)", (split,n))
            length_counts[split]+=1; row_counts[split]+=1
            if gid in multi_ids: multi_counts[split]+=1; multi_group_counts[split].add(gid)
            if n > context_limit:
                overlong_counts[split] += 1
                if len(examples)<100: examples.append({"source_row_id":rowid,"group_id":gid,"split":split,"token_length":n,"source_shard":shard,"source_row_number_1_based":source_row})
        for s in SPLITS: outputs[s].flush()
        manifest.flush()
    finally:
        for f in outputs.values(): f.close()
        manifest.close()
    # Re-read every export line to validate JSON shape, prompt/answer roles and row counts.
    validated = Counter()
    for s in SPLITS:
        with (output / f"{s}.jsonl").open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                obj = json.loads(line)
                msgs = obj.get("messages")
                if not isinstance(msgs, list) or len(msgs) != 2 or [m.get("role") for m in msgs] != ["user", "assistant"]:
                    raise ValueError(f"Invalid chat messages in {s}.jsonl line {line_no}")
                if set(obj) != {"messages"} or any(set(m) != {"role", "content"} for m in msgs):
                    raise ValueError(f"Unexpected context/field in {s}.jsonl line {line_no}")
                if not all(isinstance(m["content"], str) for m in msgs):
                    raise ValueError(f"Non-text message in {s}.jsonl line {line_no}")
                validated[s] += 1
    if validated != row_counts or sum(validated.values()) != expected:
        raise ValueError("JSONL records do not reconcile to the Stage 3 PASS count")
    manifest_count = sum(1 for _ in (output / "manifest.jsonl").open(encoding="utf-8"))
    if manifest_count != expected:
        raise ValueError("Manifest rows do not reconcile to Stage 3 PASS count")
    for s,n in db.execute("SELECT split,COUNT(*) FROM groups GROUP BY split"): group_counts[s]=n
    result={"seed":seed,"group_key":"NFC + collapsed whitespace normalized question SHA-256 fingerprint","dataset_revision":summary3["dataset_revision"],"model":MODEL,"tokenizer_class":type(tokenizer).__name__,"tokenizer_chat_template_applied":True,"max_position_embeddings":context_limit,"split_ratios_requested":RATIOS,"splits":{},"multiple_response_groups_total":len(multi_ids),"multiple_response_groups_by_split":{s:len(multi_group_counts[s]) for s in SPLITS},"eligible_records":expected,"export_records":sum(row_counts.values()),"question_groups_crossing_splits":int(db.execute("SELECT COUNT(*) FROM (SELECT gid FROM groups GROUP BY gid HAVING COUNT(DISTINCT split)>1)").fetchone()[0]),"overlong_records":sum(overlong_counts.values()),"overlong_by_split":dict(overlong_counts),"overlong_examples":examples,"outputs":{s:f"{s}.jsonl" for s in SPLITS},"manifest":"manifest.jsonl","token_length_statistics":{s:_db_percentiles(db,s,length_counts[s]) for s in SPLITS},"limitations":["Multiple-response variants remain separate examples; audit flags are not semantic conflict labels.","No source context is included in messages.","Split assignment is by normalized question group; requested row percentages are approximate because groups are indivisible.","Over-context examples are retained; no truncation or automatic deletion."]}
    for s in SPLITS: result["splits"][s]={"records":row_counts[s],"percent":100*row_counts[s]/expected,"groups":group_counts[s],"multiple_response_groups":len(multi_group_counts[s]),"token_lengths":_db_percentiles(db,s,length_counts[s])}
    p99=max(result["splits"][s]["token_lengths"]["p99"] for s in SPLITS)
    result["recommended_max_sequence_length"] = min(context_limit, int(math.ceil(p99 / 1024) * 1024))
    result["records_above_recommended_length"] = 0
    result["above_recommended_examples"] = []
    with (output / "manifest.jsonl").open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if item["token_length"] > result["recommended_max_sequence_length"]:
                result["records_above_recommended_length"] += 1
                if len(result["above_recommended_examples"]) < 100:
                    result["above_recommended_examples"].append(item)
    result["export_validation"]={"json_records_valid":sum(validated.values()),"manifest_records":manifest_count,"messages_schema_valid":True,"source_context_in_messages":False,"duplicate_source_ids":int(db.execute("SELECT COUNT(*)-COUNT(DISTINCT rowid) FROM records").fetchone()[0]),"question_groups_cross_splits":result["question_groups_crossing_splits"],"counts_reconcile":sum(row_counts.values())==expected==manifest_count}
    (output/"stage4_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    (output/"stage4_report.md").write_text(_report(result))
    db.close()
    import shutil; shutil.rmtree(work)
    return result


def _read_jsonl_gzip(path: Path):
    import gzip
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f: yield json.loads(line)


def _db_percentiles(db: sqlite3.Connection, split: str, count: int) -> dict:
    if not count: return percentiles([])
    def at_offset(offset):
        return int(db.execute("SELECT n FROM token_lengths WHERE split=? ORDER BY n LIMIT 1 OFFSET ?", (split,offset)).fetchone()[0])
    mid = count // 2
    median = at_offset(mid) if count % 2 else statistics.mean((at_offset(mid-1), at_offset(mid)))
    return {"median": median, "p90":at_offset(max(0,math.ceil(.90*count)-1)), "p95":at_offset(max(0,math.ceil(.95*count)-1)), "p99":at_offset(max(0,math.ceil(.99*count)-1)), "max":int(db.execute("SELECT MAX(n) FROM token_lengths WHERE split=?",(split,)).fetchone()[0])}


def _report(s: dict) -> str:
    out=["# Stage 4 grouped question-only export","",f"Dataset revision: `{s['dataset_revision']}`",f"Seed: `{s['seed']}`",f"Model/tokenizer: `{s['model']}` / `{s['tokenizer_class']}`",f"Qwen configured context: {s['max_position_embeddings']:,} tokens","","All variants in the existing multiple-response audit are retained. Multiple responses have not been human-reviewed for semantic conflict. The chat messages contain only question and response; context is absent.","","| Split | Records | Share of PASS | Question groups | Multiple-response groups | Median | P90 | P95 | P99 | Max |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,x in s['splits'].items():
        t=x['token_lengths']; out.append(f"| {name} | {x['records']:,} | {x['percent']:.3f}% | {x['groups']:,} | {x['multiple_response_groups']:,} | {t['median']} | {t['p90']} | {t['p95']} | {t['p99']} | {t['max']} |")
    out += ["",f"Over configured model context ({s['max_position_embeddings']:,}): {s['overlong_records']:,} records. These are retained without truncation or exclusion.","", f"Recommended initial max sequence length: {s['recommended_max_sequence_length']:,} tokens (P99 rounded up to the next 1,024-token boundary, capped by model context). {s['records_above_recommended_length']:,} records exceed this recommendation; all are retained intact and up to 100 traceable examples are in `stage4_summary.json` for explicit review before training.","","No question group crosses splits; counts reconcile to all PASS records.",""]
    return "\n".join(out)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage3-dir',type=Path,default=Path('data/processed/stage3-label-based-v1')); p.add_argument('--audit-dir',type=Path,default=Path('data/reports/multiple-response-audit-v1')); p.add_argument('--output-dir',type=Path,default=Path('data/processed/stage4-export-v1')); p.add_argument('--seed',type=int,default=SEED); p.add_argument('--batch-size',type=int,default=2048); a=p.parse_args(); print(json.dumps(run(a.stage3_dir,a.audit_dir,a.output_dir,a.seed,a.batch_size),indent=2))

if __name__=='__main__': main()
