from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import json
import logging
import os

from memory_module.embedder import BaseEmbedder, embedder_from_manifest, get_default_embedder
from memory_module.vector_store import VectorRecord, VectorStore, create_vector_store


@dataclass(frozen=True)
class RetrievalCase:
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    type: Literal["specific", "skeleton", "none"]
    cases: List[RetrievalCase]
    debug: Dict[str, Any]


CASEBANK_FORMAT_VERSION = 1


class CaseBank:
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store_factory: Optional[Callable[[str, BaseEmbedder], VectorStore]] = None,
        min_score_specific: float = 0.35,
        min_score_skeleton: float = 0.30,
        persist_dir: Optional[str] = None,
        allow_splits: Optional[List[str]] = None,
    ) -> None:
        self._embedder = embedder or get_default_embedder()
        self._vector_store_factory = vector_store_factory or create_vector_store
        self._specific = self._vector_store_factory("memento_case_specific", self._embedder)
        self._skeleton = self._vector_store_factory("memento_case_skeleton", self._embedder)
        self._min_score_specific = min_score_specific
        self._min_score_skeleton = min_score_skeleton
        self._persist_dir = Path(persist_dir) if persist_dir else _default_persist_dir()
        self._loaded = False
        self._manifest: Dict[str, Any] | None = None
        self._allow_splits = allow_splits or _default_allow_splits()

    def add_specific(self, text: str, metadata: Dict[str, Any]) -> None:
        self._specific.add_texts([text], [metadata])

    def add_skeleton(
        self,
        text: str,
        metadata: Dict[str, Any],
        forbidden_identifiers: Optional[List[str]] = None,
    ) -> None:
        if forbidden_identifiers:
            lowered = text.lower()
            for token in forbidden_identifiers:
                if token.lower() in lowered:
                    raise ValueError(f"skeleton text leaked identifier: {token}")
        self._skeleton.add_texts([text], [metadata])

    def retrieve_tiered(
        self,
        question: str,
        db_id: str,
        dialect: str,
        policy: Literal["skeleton_only", "tiered"],
        k: int = 4,
        min_score_specific: Optional[float] = None,
        min_score_skeleton: Optional[float] = None,
    ) -> RetrievalResult:
        min_specific = self._min_score_specific if min_score_specific is None else min_score_specific
        min_skeleton = self._min_score_skeleton if min_score_skeleton is None else min_score_skeleton
        debug: Dict[str, Any] = {
            "policy": policy,
            "min_score_specific": min_specific,
            "min_score_skeleton": min_skeleton,
            "specific_score": None,
            "skeleton_score": None,
            "allow_splits": list(self._allow_splits),
        }

        self._ensure_loaded()
        if policy == "skeleton_only":
            return self._retrieve_skeleton_only(question, k, min_skeleton, debug)

        if policy == "tiered":
            specific = self._query_specific(question, db_id, k)
            if specific:
                debug["specific_score"] = specific[0].score
                if specific[0].score >= min_specific:
                    debug["reason"] = "specific_hit"
                    debug["result_type"] = "specific"
                    return RetrievalResult(type="specific", cases=_to_cases(specific), debug=debug)
                debug["reason"] = "specific_below_threshold"
            else:
                debug["reason"] = "specific_empty"

            skeleton = self._query_skeleton(question, k)
            if skeleton:
                debug["skeleton_score"] = skeleton[0].score
                if skeleton[0].score >= min_skeleton:
                    debug["reason"] = "skeleton_hit"
                    debug["result_type"] = "skeleton"
                    return RetrievalResult(type="skeleton", cases=_to_cases(skeleton), debug=debug)
                debug["reason"] = "skeleton_below_threshold"

        debug.setdefault("reason", "no_match")
        debug["result_type"] = "none"
        return RetrievalResult(type="none", cases=[], debug=debug)

    def _retrieve_skeleton_only(
        self,
        question: str,
        k: int,
        min_skeleton: float,
        debug: Dict[str, Any],
    ) -> RetrievalResult:
        skeleton = self._query_skeleton(question, k)
        if skeleton:
            debug["skeleton_score"] = skeleton[0].score
            if skeleton[0].score >= min_skeleton:
                debug["reason"] = "skeleton_hit"
                debug["result_type"] = "skeleton"
                return RetrievalResult(type="skeleton", cases=_to_cases(skeleton), debug=debug)
            debug["reason"] = "skeleton_below_threshold"
        else:
            debug["reason"] = "skeleton_empty"
        debug["result_type"] = "none"
        return RetrievalResult(type="none", cases=[], debug=debug)

    def _query_specific(self, question: str, db_id: str, k: int) -> List[VectorRecord]:
        def _filter(meta: Dict[str, Any]) -> bool:
            if meta.get("db_id") != db_id:
                return False
            return self._allow_meta(meta)

        return self._specific.query(question, k=k, filter_fn=_filter)

    def _query_skeleton(self, question: str, k: int) -> List[VectorRecord]:
        return self._skeleton.query(question, k=k, filter_fn=self._allow_meta)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._persist_dir is None:
            self._loaded = True
            return
        if not self._persist_dir.exists():
            self._loaded = True
            return
        manifest = self._load_manifest()
        if manifest:
            self._apply_manifest(manifest)
        self._load_collection(self._specific, "specific")
        self._load_collection(self._skeleton, "skeleton")
        self._loaded = True

    def _load_collection(self, store: VectorStore, name: str) -> None:
        jsonl_path = self._persist_dir / f"{name}.jsonl"
        if not jsonl_path.exists():
            return
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                texts.append(record["text"])
                metadatas.append(record.get("metadata", {}))
        npy_path = self._persist_dir / f"{name}.npy"
        if npy_path.exists():
            try:
                import numpy as np

                embeddings = np.load(npy_path)
                if self._manifest:
                    expected_dim = self._manifest.get("embedder", {}).get("dim")
                    if expected_dim and embeddings.shape[1] != expected_dim:
                        message = (
                            f"CaseBank embeddings dim mismatch for {name}: "
                            f"{embeddings.shape[1]} != {expected_dim}"
                        )
                        if os.environ.get("MEMENTO_CASEBANK_STRICT") == "1":
                            raise RuntimeError(message)
                        logger.warning(message)
                        store.add_texts(texts, metadatas)
                        return
                store.add_texts_with_embeddings(texts, metadatas, embeddings)
                return
            except Exception:
                pass
        store.add_texts(texts, metadatas)

    def _load_manifest(self) -> Dict[str, Any] | None:
        manifest_path = self._persist_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return self._manifest
        except Exception:
            return None

    def _apply_manifest(self, manifest: Dict[str, Any]) -> None:
        embedder_meta = manifest.get("embedder", {})
        strict = os.environ.get("MEMENTO_CASEBANK_STRICT") == "1"
        version = manifest.get("casebank_format_version")
        if version and version != CASEBANK_FORMAT_VERSION:
            message = f"CaseBank manifest version mismatch: {version} != {CASEBANK_FORMAT_VERSION}"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
        if not embedder_meta:
            if strict:
                raise RuntimeError("Missing CaseBank manifest embedder metadata under strict mode.")
            logger.warning("CaseBank manifest missing embedder metadata.")
        else:
            requested_model = os.environ.get("MEMENTO_EMBEDDER_MODEL")
            if (
                embedder_meta.get("type") == "sentence-transformers"
                and requested_model
                and requested_model != embedder_meta.get("model")
            ):
                message = (
                    "CaseBank manifest embedder model mismatch: "
                    f"{embedder_meta.get('model')} != {requested_model}"
                )
                if strict:
                    raise RuntimeError(message)
                logger.warning(message)
            self._embedder = embedder_from_manifest(manifest)
            self._specific = self._vector_store_factory("memento_case_specific", self._embedder)
            self._skeleton = self._vector_store_factory("memento_case_skeleton", self._embedder)
        if manifest.get("similarity") != "cosine":
            message = f"CaseBank manifest similarity mismatch: {manifest.get('similarity')}"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
        if manifest.get("normalize") is not True:
            message = f"CaseBank manifest normalize mismatch: {manifest.get('normalize')}"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
        manifest_split = manifest.get("split") or manifest.get("splits")
        if manifest_split and self._allow_splits:
            allowed = set(self._allow_splits)
            manifest_splits = (
                {manifest_split} if isinstance(manifest_split, str) else set(manifest_split)
            )
            if allowed.isdisjoint(manifest_splits):
                message = (
                    "CaseBank manifest split not allowed by runtime filter: "
                    f"manifest={sorted(manifest_splits)}, allow={sorted(allowed)}"
                )
                if strict:
                    raise RuntimeError(message)
                logger.warning(message)

    def _allow_meta(self, meta: Dict[str, Any]) -> bool:
        if not self._allow_splits:
            return True
        split = meta.get("split")
        return split in self._allow_splits


def _default_persist_dir() -> Optional[Path]:
    path = os.environ.get("MEMENTO_CASEBANK_DIR")
    return Path(path) if path else None


def _default_allow_splits() -> List[str]:
    raw = os.environ.get("MEMENTO_CASEBANK_ALLOW_SPLIT")
    if not raw:
        return ["train"]
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _to_cases(records: List[VectorRecord]) -> List[RetrievalCase]:
    return [RetrievalCase(text=r.text, score=r.score, metadata=r.metadata) for r in records]


logger = logging.getLogger(__name__)
