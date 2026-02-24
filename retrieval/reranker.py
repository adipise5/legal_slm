"""
retrieval/reranker.py
BGE reranker — re-scores top-N dense candidates for final precision.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    def __init__(self):
        self._model = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        if not RERANKER_AVAILABLE:
            logger.warning("[Reranker] sentence-transformers not installed")
            self._loaded = True
            return
        try:
            logger.info("[Reranker] Loading %s ...", RERANKER_MODEL)
            self._model = CrossEncoder(RERANKER_MODEL, max_length=512)
            logger.info("[Reranker] Ready")
        except Exception as exc:
            logger.warning("[Reranker] Load failed: %s", exc)
        self._loaded = True

    def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int = 5,
    ) -> list[dict]:
        """
        Re-score candidates with cross-encoder and return top_k.
        If reranker unavailable, returns candidates[:top_k] unchanged.
        """
        if not candidates:
            return []

        self._lazy_load()

        if self._model is None:
            # No reranker — just return top_k by hybrid score
            for i, c in enumerate(candidates):
                c["rerank_score"] = c.get("_score", 0.0)
            return candidates[:top_k]

        pairs  = [(query, c["text"]) for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("[Reranker] predict failed: %s", exc)
            for i, c in enumerate(candidates):
                c["rerank_score"] = c.get("_score", 0.0)
            return candidates[:top_k]

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]