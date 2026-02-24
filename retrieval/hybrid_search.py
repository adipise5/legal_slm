"""
retrieval/hybrid_searcher.py
High-precision Hybrid Search using Qdrant Native RRF.
Scalable to 100k+ documents on Mac M4 Air.
"""
import logging
from qdrant_client import QdrantClient, models
from ingestion.embedder import Embedder
from config import QDRANT_PATH, COLLECTION_NAME, RERANK_TOP_N

logger = logging.getLogger(__name__)

class HybridSearcher:
    def __init__(self):
        # Initialize local on-disk Qdrant client
        self.client = QdrantClient(path=QDRANT_PATH)
        self.embedder = Embedder()
        
    def _create_sparse_vector(self, text: str):
        """
        Creates a simple sparse vector for exact keyword matching.
        This matches the logic used in parallel_ingestor.py.
        """
        from collections import Counter
        import re
        words = re.findall(r"\b\w+\b", text.lower())
        word_counts = Counter(words)
        
        indices = []
        values = []
        for word, count in word_counts.items():
            # Standard hashing for portable, disk-based sparse indexing
            idx = hash(word) % 32768
            indices.append(idx)
            values.append(float(count))
            
        return models.SparseVector(indices=indices, values=values)

    def hybrid_search(self, query: str, top_n: int = 50, doc_type_filter: str = None):
        """
        Performs Hybrid Search (Dense + Sparse) with Native RRF Fusion.
        """
        # 1. Encode query for both dense and sparse
        query_dense = self.embedder.encode([f"search_query: {query}"])[0].tolist()
        query_sparse = self._create_sparse_vector(query)

        # 2. Define Metadata Filter
        query_filter = None
        if doc_type_filter:
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="doc_type", 
                    match=models.MatchValue(value=doc_type_filter)
                )]
            )

        # 3. Perform Universal Query with RRF
        # This executes two sub-queries and fuses them into a single ranked list
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="text-dense",
                    limit=top_n,
                    filter=query_filter
                ),
                models.Prefetch(
                    query=query_sparse,
                    using="text-sparse",
                    limit=top_n,
                    filter=query_filter
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_n,
        )

        # 4. Transform to standard candidate format
        candidates = []
        for point in response.points:
            candidates.append({
                "id": point.id,
                "text": point.payload.get("text", ""),
                "source_file": point.payload.get("source_file", "Unknown"),
                "page_number": point.payload.get("page_number", 0),
                "doc_type": point.payload.get("doc_type", "Unknown"),
                "score": point.score,
                "search_type": "hybrid_rrf"
            })
            
        return candidates