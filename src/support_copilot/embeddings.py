from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List

import numpy as np
from langchain_core.embeddings import Embeddings


class LocalHashEmbeddings(Embeddings):
    """Small, dependency-light embeddings that work offline.

    This is NOT SOTA semantic embeddings; it's a deterministic feature-hashing
    baseline that is fast, stable, and good enough for a minimal RAG demo.
    """

    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[\w]+", (text or "").lower())

    def _hash_token(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False)

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokenize(text):
            idx = self._hash_token(tok) % self.dim
            vec[idx] += 1.0

        norm = float(np.linalg.norm(vec))
        if math.isfinite(norm) and norm > 0:
            vec /= norm
        return vec.astype(np.float32).tolist()

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def create_embeddings() -> Embeddings:
    """Factory used by the rest of the codebase."""
    return LocalHashEmbeddings()