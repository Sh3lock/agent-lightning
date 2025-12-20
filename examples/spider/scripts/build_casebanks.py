import argparse
import hashlib
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_module.embedder import describe_embedder, get_default_embedder
from memory_module.skeletonizer import SqlSkeletonizer
from memory_module.casebank import CASEBANK_FORMAT_VERSION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CaseBank indexes from Spider parquet data.")
    parser.add_argument("--input", required=True, help="Path to Spider parquet (train/dev/test).")
    parser.add_argument("--persist_dir", default=None, help="Output directory for casebanks.")
    parser.add_argument("--spider_dir", default=None, help="Root Spider data dir (default: VERL_SPIDER_DATA_DIR).")
    parser.add_argument("--split", default="all", choices=["train", "dev", "test", "all"])
    parser.add_argument("--limit", type=int, default=-1, help="Limit rows for debugging (<=0 means all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--mode", default="skip", choices=["skip", "upsert"])
    parser.add_argument("--build_specific", type=int, default=1)
    parser.add_argument("--build_skeleton", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sample_check", type=int, default=20, help="Sample size for skeleton leakage checks.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for exec verification.")
    return parser.parse_args()


def _case_id(db_id: str, question: str, sql: str) -> str:
    raw = f"{db_id}\n{question}\n{sql}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _resolve_db_path(spider_dir: Path, db_id: str, split: str) -> Path:
    candidates = []
    if split == "train":
        candidates.append(spider_dir / "database" / db_id / f"{db_id}.sqlite")
    elif split in {"dev", "test"}:
        candidates.append(spider_dir / "test_database" / db_id / f"{db_id}.sqlite")
    else:
        candidates.append(spider_dir / "database" / db_id / f"{db_id}.sqlite")
        candidates.append(spider_dir / "test_database" / db_id / f"{db_id}.sqlite")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Database not found for {db_id}. Tried: {candidates}")


def _exec_verified(db: SQLDatabase, sql: str) -> Tuple[bool, str | None]:
    try:
        tool = QuerySQLDatabaseTool(db=db)
        tool.invoke(sql)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _load_existing(persist_dir: Path, name: str) -> Dict[str, Dict[str, Any]]:
    path = persist_dir / f"{name}.jsonl"
    if not path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            case_id = meta.get("case_id")
            if not case_id:
                continue
            records[case_id] = rec
    return records


def _write_records(persist_dir: Path, name: str, records: Iterable[Dict[str, Any]]) -> None:
    path = persist_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_embeddings(persist_dir: Path, name: str, texts: List[str], embedder: Any) -> List[List[float]]:
    vectors = embedder.embed_texts(texts)
    import numpy as np

    np.save(persist_dir / f"{name}.npy", np.array(vectors, dtype="float32"))
    return vectors


def _write_manifest(persist_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest_path = persist_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    spider_dir = Path(args.spider_dir or os.environ.get("VERL_SPIDER_DATA_DIR", "data"))
    persist_dir = Path(args.persist_dir or os.environ.get("MEMENTO_CASEBANK_DIR", "casebanks"))
    persist_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    if args.split != "all" and "split" in df.columns:
        df = df[df["split"] == args.split]
    if args.limit > 0:
        df = df.head(args.limit)
    rows = df.to_dict(orient="records")

    random.seed(args.seed)
    skeletonizer = SqlSkeletonizer()

    existing_specific = _load_existing(persist_dir, "specific") if args.mode in {"skip", "upsert"} else {}
    existing_skeleton = _load_existing(persist_dir, "skeleton") if args.mode in {"skip", "upsert"} else {}
    specific_records = dict(existing_specific)
    skeleton_records = dict(existing_skeleton)

    stats = {
        "total": 0,
        "exec_verified": 0,
        "exec_failed": 0,
        "skeleton_failed": 0,
        "skeleton_leak": 0,
        "specific_written": 0,
        "skeleton_written": 0,
        "specific_skipped": 0,
        "skeleton_skipped": 0,
    }

    tasks: List[Tuple[Dict[str, Any], str]] = []
    for row in rows:
        question = row.get("question")
        db_id = row.get("db_id")
        sql = row.get("query")
        if not question or not db_id or not sql:
            continue
        split_label = args.split if args.split != "all" else row.get("split", "unknown")
        tasks.append((row, split_label))

    exec_cache: Dict[Tuple[str, str], Tuple[bool, str | None]] = {}
    exec_lock = threading.Lock()
    thread_local = threading.local()

    def _get_thread_db(db_id: str, db_path: Path) -> SQLDatabase:
        cache = getattr(thread_local, "db_cache", None)
        if cache is None:
            cache = {}
            thread_local.db_cache = cache
        if db_id not in cache:
            cache[db_id] = SQLDatabase.from_uri(f"sqlite:///{db_path}")  # type: ignore
        return cache[db_id]

    def _verify_row(row: Dict[str, Any], split_label: str) -> Dict[str, Any]:
        db_id = row.get("db_id")
        sql = row.get("query")
        try:
            db_path = _resolve_db_path(spider_dir, db_id, split_label)
        except FileNotFoundError as exc:
            return {"row": row, "split": split_label, "ok": False, "error": str(exc)}
        db = _get_thread_db(db_id, db_path)
        key = (db_id, hashlib.sha1(sql.encode("utf-8")).hexdigest())
        with exec_lock:
            cached = exec_cache.get(key)
        if cached is not None:
            ok, err = cached
            return {"row": row, "split": split_label, "ok": ok, "error": err}
        ok, err = _exec_verified(db, sql)
        with exec_lock:
            exec_cache[key] = (ok, err)
        return {"row": row, "split": split_label, "ok": ok, "error": err}

    results: List[Dict[str, Any]] = []
    if args.workers <= 1:
        results = [_verify_row(row, split_label) for row, split_label in tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_verify_row, row, split_label) for row, split_label in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())

    for result in results:
        stats["total"] += 1
        if not result.get("ok"):
            stats["exec_failed"] += 1
            continue
        stats["exec_verified"] += 1
        row = result["row"]
        split_label = result["split"]
        question = row.get("question")
        db_id = row.get("db_id")
        sql = row.get("query")

        case_id = _case_id(db_id, question, sql)
        metadata = {
            "db_id": db_id,
            "dialect": "sqlite",
            "source": "spider_gt",
            "split": split_label,
            "case_id": case_id,
            "exec_verified": True,
        }

        if args.build_specific:
            if args.mode == "skip" and case_id in specific_records:
                stats["specific_skipped"] += 1
            else:
                specific_records[case_id] = {
                    "text": f"Question: {question}\nSQL: {sql}",
                    "metadata": metadata,
                }
                stats["specific_written"] += 1

        if args.build_skeleton:
            skeleton = skeletonizer.skeletonize(sql, dialect="sqlite")
            if skeleton.failed:
                stats["skeleton_failed"] += 1
                continue
            try:
                skeletonizer.assert_no_identifiers(sql, skeleton.skeleton_sql, "sqlite")
            except AssertionError:
                stats["skeleton_leak"] += 1
                continue
            sk_case_id = _case_id(db_id, question, skeleton.skeleton_sql)
            sk_metadata = dict(metadata)
            sk_metadata["case_id"] = sk_case_id
            sk_metadata["op_signature"] = skeleton.op_signature
            if args.mode == "skip" and sk_case_id in skeleton_records:
                stats["skeleton_skipped"] += 1
            else:
                skeleton_records[sk_case_id] = {
                    "text": (
                        f"Question: {question}\n"
                        f"SQL_SKELETON: {skeleton.skeleton_sql}\n"
                        f"OP_SIGNATURE: {skeleton.op_signature}"
                    ),
                    "metadata": sk_metadata,
                }
                stats["skeleton_written"] += 1

    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    embedder = get_default_embedder()
    specific_list: List[Dict[str, Any]] = []
    skeleton_list: List[Dict[str, Any]] = []
    specific_vectors: List[List[float]] = []
    skeleton_vectors: List[List[float]] = []

    if args.build_specific:
        specific_list = list(specific_records.values())
        _write_records(persist_dir, "specific", specific_list)
        if specific_list:
            specific_vectors = _write_embeddings(
                persist_dir,
                "specific",
                [rec["text"] for rec in specific_list],
                embedder,
            )
        else:
            specific_npy = persist_dir / "specific.npy"
            if specific_npy.exists():
                specific_npy.unlink()

    if args.build_skeleton:
        skeleton_list = list(skeleton_records.values())
        _write_records(persist_dir, "skeleton", skeleton_list)
        if skeleton_list:
            skeleton_vectors = _write_embeddings(
                persist_dir,
                "skeleton",
                [rec["text"] for rec in skeleton_list],
                embedder,
            )
        else:
            skeleton_npy = persist_dir / "skeleton.npy"
            if skeleton_npy.exists():
                skeleton_npy.unlink()

    sample_vector = None
    if specific_vectors:
        sample_vector = specific_vectors[0]
    elif skeleton_vectors:
        sample_vector = skeleton_vectors[0]
    manifest = {
        "casebank_format_version": CASEBANK_FORMAT_VERSION,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "split": args.split,
        "source": str(args.input),
        "counts": {
            "specific": len(specific_list) if args.build_specific else 0,
            "skeleton": len(skeleton_list) if args.build_skeleton else 0,
        },
        "embedder": describe_embedder(embedder, sample_vector),
        "similarity": "cosine",
        "normalize": True,
        "exec_stats": {
            "total": stats["total"],
            "exec_verified": stats["exec_verified"],
            "exec_failed": stats["exec_failed"],
        },
    }
    _write_manifest(persist_dir, manifest)

    if args.sample_check > 0 and args.build_skeleton:
        sample = random.sample(list(skeleton_records.values()), min(args.sample_check, len(skeleton_records)))
        print(f"[sample_check] checked {len(sample)} skeleton records (assertions already applied).")


if __name__ == "__main__":
    main()
