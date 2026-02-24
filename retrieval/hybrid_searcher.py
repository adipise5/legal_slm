"""
retrieval/hybrid_searcher.py
Dense (Qdrant cosine) + BM25 hybrid search over ingested legal documents.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import (
    COLLECTION_NAME, QDRANT_PATH, BM25_INDEX_PATH,
    EMBED_MODEL, RERANK_TOP_N,
)
from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

# Weights for hybrid score: final = DENSE_W × dense + BM25_W × bm25
DENSE_W = 0.65
BM25_W  = 0.35


class HybridSearcher:
    def __init__(self):
        self.qdrant   = QdrantClient(path=QDRANT_PATH)
        self.embedder = Embedder()
        self._bm25    = None
        self._bm25_docs: list[dict] = []
        self._load_bm25()

    def _load_bm25(self):
        p = Path(BM25_INDEX_PATH)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    data = pickle.load(f)
                self._bm25      = data.get("index")
                self._bm25_docs = data.get("docs", [])
                logger.info("[BM25] Loaded %d docs", len(self._bm25_docs))
            except Exception as exc:
                logger.warning("[BM25] Load failed: %s", exc)

    def hybrid_search(
        self,
        query:           str,
        top_n:           int  = RERANK_TOP_N,
        doc_type_filter: str | None = None,
    ) -> list[dict]:
        """
        Returns up to top_n candidate dicts, sorted by hybrid score.
        Each dict has: text, source_file, page_number, doc_type,
                       chunk_index, ocr_confidence, _score.
        Returns [] if index is empty.
        """
        # ── Check index has data ────────────────────────────────
        try:
            vec_count = self.qdrant.count(COLLECTION_NAME).count
        except Exception:
            vec_count = 0

        if vec_count == 0:
            logger.warning("[Search] Qdrant collection is empty — run ingestion first")
            return []

        # ── Embed query ────────────────────────────────────────
        query_vec = self.embedder.encode(
            [f"search_query: {query}"]
        )[0].tolist()

        # ── Dense search ───────────────────────────────────────
        qdrant_filter = None
        if doc_type_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="doc_type",
                    match=MatchValue(value=doc_type_filter),
                )]
            )

        try:
            dense_hits = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vec,
                limit=top_n * 2,   # over-fetch before BM25 fusion
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("[Search] Qdrant search failed: %s", exc)
            return []

        # Build id→score map for dense
        dense_map: dict[int, float] = {}
        doc_map:   dict[int, dict]  = {}
        for hit in dense_hits:
            dense_map[hit.id] = hit.score
            doc_map[hit.id]   = {
                "text":           hit.payload.get("text", ""),
                "source_file":    hit.payload.get("source_file", ""),
                "page_number":    hit.payload.get("page_number", 0),
                "doc_type":       hit.payload.get("doc_type", "Unknown"),
                "chunk_index":    hit.payload.get("chunk_index", 0),
                "ocr_confidence": hit.payload.get("ocr_confidence", 1.0),
            }

        # ── BM25 search ────────────────────────────────────────
        bm25_map: dict[int, float] = {}
        if self._bm25 and self._bm25_docs:
            tokens  = query.lower().split()
            scores  = self._bm25.get_scores(tokens)
            # Normalise BM25 scores to [0,1]
            max_s   = scores.max() if scores.max() > 0 else 1.0
            for i, s in enumerate(scores):
                if s > 0 and i < len(self._bm25_docs):
                    doc_id = self._bm25_docs[i].get("id")
                    if doc_id is not None:
                        bm25_map[doc_id] = float(s / max_s)
                        if doc_id not in doc_map:
                            doc_map[doc_id] = {
                                "text":           self._bm25_docs[i].get("text", ""),
                                "source_file":    self._bm25_docs[i].get("source_file", ""),
                                "page_number":    self._bm25_docs[i].get("page_number", 0),
                                "doc_type":       self._bm25_docs[i].get("doc_type", "Unknown"),
                                "chunk_index":    0,
                                "ocr_confidence": 1.0,
                            }

        # ── Hybrid fusion ──────────────────────────────────────
        all_ids = set(dense_map) | set(bm25_map)
        results = []
        for doc_id in all_ids:
            d_score = dense_map.get(doc_id, 0.0)
            b_score = bm25_map.get(doc_id, 0.0)
            hybrid  = DENSE_W * d_score + BM25_W * b_score
            entry   = dict(doc_map[doc_id])
            entry["_score"] = hybrid
            results.append(entry)

        results.sort(key=lambda x: x["_score"], reverse=True)
        return results[:top_n]