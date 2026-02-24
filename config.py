"""
Central configuration — change hardware targets, paths, and 
model choices here without touching pipeline code.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PDF_DIR          = Path("./data/legal_pdfs")
HASH_DB_PATH     = Path("./state/processed_hashes.json")
BM25_INDEX_PATH  = Path("./state/bm25_index.pkl")

# ── Qdrant ─────────────────────────────────────────────────────────
QDRANT_PATH      = "./state/qdrant_db"   # Local on-disk mode (no server needed)
COLLECTION_NAME  = "legal_docs"

# ── Embedding model (runs fully locally via sentence-transformers) ──
# nomic-embed-text: 768-dim, Apache 2.0, strong on long documents
EMBED_MODEL      = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIMENSION  = 768

# ── Reranker ───────────────────────────────────────────────────────
RERANKER_MODEL   = "BAAI/bge-reranker-v2-m3"  # Best multilingual reranker
RERANK_TOP_N     = 50   # Pull this many from Qdrant before reranking
FINAL_TOP_K      = 5    # Final chunks sent to LLM

# ── Chunking ───────────────────────────────────────────────────────
CHUNK_SIZE       = 512   # tokens — good for legal paragraphs
CHUNK_OVERLAP    = 64    # preserve cross-sentence context

# ── LLM (via Ollama) ───────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
LLM_MODEL        = "mistral:7b-instruct-q4_K_M"
LLM_TEMPERATURE  = 0.0   # Deterministic for legal accuracy
LLM_CTX_WINDOW   = 8192