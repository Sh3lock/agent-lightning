from __future__ import annotations

import math
import os
import re
from typing import Callable, List


class BaseEmbedder:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
            idx = hash(token) % self.dim
            vec[idx] += 1.0
        return _l2_normalize(vec)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class LazyEmbedder(BaseEmbedder):
    def __init__(self, factory: Callable[[], BaseEmbedder]) -> None:
        self._factory = factory
        self._embedder: BaseEmbedder | None = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._embedder is None:
            self._embedder = self._factory()
        return self._embedder.embed_texts(texts)

    def resolve(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = self._factory()
        return self._embedder


def get_default_embedder() -> LazyEmbedder:
    def _factory() -> BaseEmbedder:
        model_name = os.environ.get("MEMENTO_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:
            return HashingEmbedder()

    return LazyEmbedder(_factory)


def describe_embedder(embedder: BaseEmbedder, sample_vector: List[float] | None = None) -> dict:
    resolved = embedder
    if isinstance(embedder, LazyEmbedder):
        resolved = embedder.resolve()
    meta = {"normalize": True}
    if isinstance(resolved, HashingEmbedder):
        meta.update({"type": "hashing", "dim": resolved.dim})
    elif isinstance(resolved, SentenceTransformerEmbedder):
        meta.update({"type": "sentence-transformers", "model": resolved.model_name})
    if sample_vector is not None:
        meta["dim"] = len(sample_vector)
    return meta


def embedder_from_manifest(manifest: dict) -> BaseEmbedder:
    embedder_meta = manifest.get("embedder", {})
    embedder_type = embedder_meta.get("type", "hashing")
    if embedder_type == "sentence-transformers":
        model_name = embedder_meta.get("model", "all-MiniLM-L6-v2")
        def _factory() -> BaseEmbedder:
            try:
                return SentenceTransformerEmbedder(model_name)
            except Exception:
                return HashingEmbedder(dim=int(embedder_meta.get("dim", 128)))
        return LazyEmbedder(_factory)
    return HashingEmbedder(dim=int(embedder_meta.get("dim", 128)))


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]
