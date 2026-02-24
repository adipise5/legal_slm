"""
ingestion/ingestor.py

Core ingestion utilities:
  - Hash-based incremental detection (relative paths — cross-platform)
  - Qdrant collection initialisation
  - BM25 index persistence
  - Single-process ingestion (used as fallback / testing)

Used by parallel_ingestor.py for shared helpers.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import (
    BM25_INDEX_PATH,
    COLLECTION_NAME,
    EMBED_DIMENSION,
    HASH_DB_PATH,
    PDF_DIR,
    QDRANT_PATH,
)

# ────────────────────────────────────────────────────────────────────
# File hashing
# ────────────────────────────────────────────────────────────────────

def compute_file_hash(filepath: str) -> str:
    """SHA-256 hash of file contents. Detects both new and modified files."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ────────────────────────────────────────────────────────────────────
# Hash ledger (relative-path keys for portability)
# ────────────────────────────────────────────────────────────────────

def load_hash_db() -> dict:
    """Load the ledger of previously processed file hashes."""
    Path(HASH_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    p = Path(HASH_DB_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_hash_db(db: dict) -> None:
    """Persist the hash ledger atomically (write-then-rename)."""
    p    = Path(HASH_DB_PATH)
    tmp  = p.with_suffix(".tmp")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)   # atomic on POSIX; near-atomic on Windows


# ────────────────────────────────────────────────────────────────────
# Qdrant
# ────────────────────────────────────────────────────────────────────

def init_qdrant() -> QdrantClient:
    """
    Initialise Qdrant in local on-disk mode.
    Creates the collection if it does not exist.
    Payload indexes are skipped (no-op in local mode; use server mode
    for high-volume filtered queries).
    """
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBED_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Created collection '{COLLECTION_NAME}'")
    else:
        count = client.count(COLLECTION_NAME).count
        print(f"[Qdrant] Collection '{COLLECTION_NAME}' exists with {count:,} vectors")

    return client


# ────────────────────────────────────────────────────────────────────
# BM25 index
# ────────────────────────────────────────────────────────────────────

def load_bm25_index():
    """
    Load BM25 index and document store from disk.
    Returns (BM25Okapi | None, list[dict]).
    """
    p = Path(BM25_INDEX_PATH)
    if p.exists():
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            return data.get("index"), data.get("docs", [])
        except Exception:
            return None, []
    return None, []


def save_bm25_index(index, docs: list[dict]) -> None:
    """Persist BM25 index and document store."""
    p = Path(BM25_INDEX_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"index": index, "docs": docs}, f)
    tmp.replace(p)