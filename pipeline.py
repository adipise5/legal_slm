"""
pipeline.py

Query flow:
  1. Hybrid search (dense + BM25) in the local 26k-doc index
  2. BGE reranker picks the best chunks
  3. Mistral 7B generates a grounded answer with citations

  If no documents match → Mistral gives a general AI answer
  (clearly labelled, no document citations, web refs OK from model knowledge)
"""
from __future__ import annotations

import logging

from retrieval.hybrid_searcher import HybridSearcher
from retrieval.reranker import BGEReranker
from generation.llm_chain import LegalLLMChain
from config import RERANK_TOP_N, FINAL_TOP_K

logger = logging.getLogger(__name__)


class LegalSLMPipeline:

    def __init__(self):
        self.searcher = HybridSearcher()
        self.reranker = BGEReranker()
        self.llm      = LegalLLMChain()

    def query(
        self,
        question: str,
        doc_type_filter: str | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Returns a dict with keys:
          query        - original question
          answer       - text answer (may include citations)
          sources      - list of source dicts (empty for AI fallback)
          answer_type  - "document_grounded" | "general_ai_fallback"
        """
        if verbose:
            print(f"[Pipeline] Query: {question!r}  filter={doc_type_filter}")

        # ── 1. Retrieve candidates ────────────────────────────────
        candidates = self.searcher.hybrid_search(
            query=question,
            top_n=RERANK_TOP_N,
            doc_type_filter=doc_type_filter,
        )

        if verbose:
            print(f"[Pipeline] Hybrid search returned {len(candidates)} candidates")

        # ── 2. No documents → general AI fallback ────────────────
        if not candidates:
            logger.info("[Pipeline] No document matches — using general AI fallback")
            return self.llm.generate_general_ai(question)

        # ── 3. Rerank ─────────────────────────────────────────────
        top_chunks = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=FINAL_TOP_K,
        )

        if verbose:
            for c in top_chunks:
                print(f"  #{c['rank']} {c['source_file']} p{c['page_number']} "
                      f"score={c.get('rerank_score', 0):.4f}")

        # ── 4. Generate grounded answer ───────────────────────────
        return self.llm.generate(question, top_chunks)


if __name__ == "__main__":
    import json
    pipeline = LegalSLMPipeline()
    result   = pipeline.query("What are the charges in the FIR?", verbose=True)
    print("\n" + "═" * 60)
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['rank']}] {s['source_file']} p{s['page_number']} "
              f"({s['doc_type']}) score={s['rerank_score']}")