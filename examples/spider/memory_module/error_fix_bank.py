from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from memory_module.embedder import BaseEmbedder, get_default_embedder
from memory_module.vector_store import VectorRecord, VectorStore, create_vector_store


@dataclass(frozen=True)
class FixHint:
    text: str
    score: float
    metadata: Dict[str, Any]


class ErrorFixBank:
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store_factory: Optional[Callable[[str, BaseEmbedder], VectorStore]] = None,
        min_score: float = 0.30,
    ) -> None:
        self._embedder = embedder or get_default_embedder()
        self._vector_store_factory = vector_store_factory or create_vector_store
        self._store = self._vector_store_factory("memento_error_fix", self._embedder)
        self._min_score = min_score
        self._seeded = False
        self._seed_hints: List[Dict[str, Any]] = [
            {
                "text": "MissingColumn: Inspect CURRENT SCHEMA for the closest column name; add table alias or prefix if needed.",
                "metadata": {"error_type": "MissingColumn", "dialect": "any"},
            },
            {
                "text": "MissingTable: Verify table names in CURRENT SCHEMA; check pluralization and whether a JOIN to another table is required.",
                "metadata": {"error_type": "MissingTable", "dialect": "any"},
            },
            {
                "text": "AmbiguousColumn: Qualify the column with the correct table alias to remove ambiguity.",
                "metadata": {"error_type": "AmbiguousColumn", "dialect": "any"},
            },
            {
                "text": "SyntaxError: Locate the reported token; check comma placement, parentheses balance, and keyword order.",
                "metadata": {"error_type": "SyntaxError", "dialect": "any"},
            },
        ]

    def add_fix_hint(self, text: str, metadata: Dict[str, Any]) -> None:
        self._ensure_seeded()
        self._store.add_texts([text], [metadata])

    def retrieve_fix_hints(
        self,
        error_type: str,
        dialect: str,
        query_text: str,
        k: int = 4,
        min_score: Optional[float] = None,
    ) -> List[FixHint]:
        self._ensure_seeded()
        threshold = self._min_score if min_score is None else min_score

        def _filter(meta: Dict[str, Any]) -> bool:
            type_ok = meta.get("error_type") in {error_type, "any"}
            dialect_ok = meta.get("dialect") in {dialect, "any"}
            return type_ok and dialect_ok

        records = self._store.query(query_text, k=k, filter_fn=_filter)
        hints = [
            FixHint(text=rec.text, score=rec.score, metadata=rec.metadata)
            for rec in records
            if rec.score >= threshold
        ]
        return hints

    def _ensure_seeded(self) -> None:
        if self._seeded:
            return
        for hint in self._seed_hints:
            self._store.add_texts([hint["text"]], [hint["metadata"]])
        self._seeded = True
