from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

from memory_module.embedder import BaseEmbedder


@dataclass(frozen=True)
class VectorRecord:
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    def add_texts(self, texts: Sequence[str], metadatas: Sequence[dict[str, Any]]) -> None:
        raise NotImplementedError

    def add_texts_with_embeddings(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Any,
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        query_text: str,
        k: int,
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> List[VectorRecord]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self, name: str, embedder: BaseEmbedder) -> None:
        self.name = name
        self._embedder = embedder
        self._texts: List[str] = []
        self._metadatas: List[dict[str, Any]] = []
        self._vectors: List[List[float]] = []

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[dict[str, Any]]) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")
        embeddings = [_normalize_vector(vec) for vec in self._embedder.embed_texts(list(texts))]
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._vectors.extend(embeddings)

    def add_texts_with_embeddings(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Any,
    ) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")
        normalized = [_normalize_vector(vec) for vec in embeddings]
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._vectors.extend(normalized)

    def query(
        self,
        query_text: str,
        k: int,
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> List[VectorRecord]:
        if not self._texts:
            return []
        query_vec = _normalize_vector(self._embedder.embed_texts([query_text])[0])
        scored: List[VectorRecord] = []
        for text, meta, vec in zip(self._texts, self._metadatas, self._vectors):
            if filter_fn and not filter_fn(meta):
                continue
            score = _dot(vec, query_vec)
            scored.append(VectorRecord(text=text, score=score, metadata=meta))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


class FaissVectorStore(VectorStore):
    def __init__(self, name: str, embedder: BaseEmbedder, dim: int, index: Any) -> None:
        self.name = name
        self._embedder = embedder
        self._index = index
        self._dim = dim
        self._texts: List[str] = []
        self._metadatas: List[dict[str, Any]] = []

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[dict[str, Any]]) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")
        embeddings = self._embedder.embed_texts(list(texts))
        import numpy as np

        arr = _normalize_numpy(np.array(embeddings, dtype="float32"))
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError("embedding dimension mismatch for FAISS store")
        self._index.add(arr)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def add_texts_with_embeddings(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Any,
    ) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")
        import numpy as np

        arr = _normalize_numpy(np.array(embeddings, dtype="float32"))
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError("embedding dimension mismatch for FAISS store")
        self._index.add(arr)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def query(
        self,
        query_text: str,
        k: int,
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> List[VectorRecord]:
        if not self._texts:
            return []
        embeddings = self._embedder.embed_texts([query_text])
        import numpy as np

        arr = _normalize_numpy(np.array(embeddings, dtype="float32"))
        scores, indices = self._index.search(arr, k)
        scored: List[VectorRecord] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._texts):
                continue
            meta = self._metadatas[idx]
            if filter_fn and not filter_fn(meta):
                continue
            scored.append(VectorRecord(text=self._texts[idx], score=float(score), metadata=meta))
        return scored


class LazyVectorStore(VectorStore):
    def __init__(self, name: str, embedder: BaseEmbedder) -> None:
        self.name = name
        self._embedder = embedder
        self._inner: VectorStore | None = None

    def add_texts(self, texts: Sequence[str], metadatas: Sequence[dict[str, Any]]) -> None:
        self._ensure_inner_for_add().add_texts(texts, metadatas)

    def add_texts_with_embeddings(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Any,
    ) -> None:
        inner = self._ensure_inner_for_add()
        if hasattr(inner, "add_texts_with_embeddings"):
            inner.add_texts_with_embeddings(texts, metadatas, embeddings)
        else:
            inner.add_texts(texts, metadatas)

    def query(
        self,
        query_text: str,
        k: int,
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> List[VectorRecord]:
        if self._inner is None:
            return []
        return self._inner.query(query_text, k=k, filter_fn=filter_fn)

    def _ensure_inner_for_add(self) -> VectorStore:
        if self._inner is not None:
            return self._inner
        self._inner = _create_inner_store(self.name, self._embedder)
        return self._inner


def create_vector_store(name: str, embedder: BaseEmbedder) -> VectorStore:
    # Keep initialization lightweight: do not embed or load models here.
    # The concrete store is selected on first add (or later extension).
    return LazyVectorStore(name, embedder)


def _create_inner_store(name: str, embedder: BaseEmbedder) -> VectorStore:
    try:
        import faiss  # type: ignore
    except Exception:
        return InMemoryVectorStore(name, embedder)

    probe_vec = embedder.embed_texts(["probe"])
    dim = len(probe_vec[0]) if probe_vec and probe_vec[0] is not None else 0
    if dim <= 0:
        return InMemoryVectorStore(name, embedder)
    index = faiss.IndexFlatIP(dim)
    return FaissVectorStore(name, embedder, dim, index)


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _normalize_vector(vec: Iterable[float]) -> List[float]:
    values = list(vec)
    norm = sum(x * x for x in values) ** 0.5
    if norm == 0.0:
        return values
    return [x / norm for x in values]


def _normalize_numpy(arr: "Any") -> "Any":
    import numpy as np

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms
