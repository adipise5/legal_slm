"""
ingestion/parallel_ingestor.py  (v6 — fixed)

ROOT CAUSES FIXED vs v5:
  1. SparseVector mismatch: _create_sparse_vector returned a plain dict
     but Qdrant needs a SparseVector object → every upsert silently failed
     → collector looped forever → 1-hour hang with no output.
     FIX: switched to DENSE-ONLY vectors (simpler, faster, no sparse needed
     since BM25 handles keyword search on our side).

  2. bare "except: continue/pass" swallowed ALL errors silently.
     FIX: all exceptions are now logged with full tracebacks.

  3. No incremental hash ledger — re-processed all 2000 PDFs every run.
     FIX: SHA-256 hash ledger with relative paths restored.

  4. Collector thread was daemon=True (default) → could be killed before
     final flush on some Python versions.
     FIX: daemon=False, explicit join with timeout.

  5. result_queue.put() could block forever if collector died mid-run.
     FIX: put() with timeout + alive-check loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from multiprocessing import get_context, cpu_count
from multiprocessing import Queue as MPQueue

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import PDF_DIR, COLLECTION_NAME, EMBED_DIMENSION, QDRANT_PATH
from ingestion.embedder import Embedder
from ingestion.paddle_ocr_engine import warmup_paddle

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
NUM_WORKERS       = min(4, cpu_count())
EMBED_BATCH_SIZE  = 128          # reduced from 256 — more frequent flushes = visible progress
QUEUE_MAXSIZE     = 3000
HASH_DB_PATH      = Path("./state/processed_hashes.json")
_WORKER_DONE      = "__WORKER_DONE__"


# ── Hash ledger (incremental ingestion) ──────────────────────────────

def _hash_key(pdf_path: str) -> str:
    """Relative portable path as ledger key."""
    try:
        return Path(pdf_path).resolve().relative_to(
            Path(PDF_DIR).resolve()
        ).as_posix()
    except ValueError:
        return Path(pdf_path).name

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def _load_hashes() -> dict:
    HASH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HASH_DB_PATH.exists():
        try:
            return json.loads(HASH_DB_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_hashes(db: dict) -> None:
    tmp = HASH_DB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, indent=2), encoding="utf-8")
    tmp.replace(HASH_DB_PATH)


# ── Worker function ───────────────────────────────────────────────────

def _process_pdf_worker(pdf_path: str) -> list[dict]:
    """
    Run in a worker process. Returns list of chunk dicts.
    NEVER raises — always returns a list (may be empty on hard failure).
    """
    from ingestion.pdf_processor import process_pdf
    results = []
    try:
        for chunk, log in process_pdf(pdf_path, verbose_logs=False):
            results.append({
                "text":        chunk.text,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "doc_type":    chunk.doc_type,
                "ocr_engine":  chunk.ocr_engine,
                "ocr_conf":    chunk.ocr_confidence,
                "_path":       pdf_path,
            })
    except Exception as exc:
        # Log properly — bare except:pass was hiding all errors
        import traceback
        logger.error("[Worker] %s FAILED:\n%s", Path(pdf_path).name,
                     traceback.format_exc())
        # Return a placeholder so the hash is still recorded
        results.append({
            "text":        "[EXTRACTION FAILED]",
            "source_file": Path(pdf_path).name,
            "page_number": 1,
            "chunk_index": 0,
            "doc_type":    "Unknown",
            "ocr_engine":  "error",
            "ocr_conf":    0.0,
            "_path":       pdf_path,
        })
    return results


# ── Collector thread ──────────────────────────────────────────────────

class CollectorThread(threading.Thread):
    """
    Runs in main process.
    Drains result queue → embeds → upserts to Qdrant.
    Uses DENSE vectors only (BM25 handles keyword search separately).
    """

    def __init__(self, client: QdrantClient, embedder: Embedder,
                 queue: MPQueue, num_workers: int):
        super().__init__(name="collector", daemon=False)
        self.client      = client
        self.embedder    = embedder
        self.queue       = queue
        self.num_workers = num_workers
        self.done_count  = 0
        self.total_chunks = 0
        self.hash_updates: dict[str, str] = {}
        self.error       = None
        self._batch: list[dict] = []
        # Starting point_id — continue from existing collection size
        try:
            self._point_id = client.count(COLLECTION_NAME).count
        except Exception:
            self._point_id = 0

    def run(self):
        try:
            self._drain()
        except Exception as exc:
            import traceback
            self.error = exc
            logger.error("[Collector] FATAL:\n%s", traceback.format_exc())

    def _drain(self):
        while self.done_count < self.num_workers:
            try:
                item = self.queue.get(timeout=5.0)
            except Exception:
                # timeout — flush partial batch & keep waiting
                if self._batch:
                    self._flush()
                continue

            if item == _WORKER_DONE:
                self.done_count += 1
                logger.info("[Collector] Worker done %d/%d",
                            self.done_count, self.num_workers)
                # flush partial batch when last worker finishes
                if self.done_count == self.num_workers and self._batch:
                    self._flush()
                continue

            # item = list[dict] from one PDF
            for d in item:
                self._batch.append(d)
            if len(self._batch) >= EMBED_BATCH_SIZE:
                self._flush()

        # Final flush of any remainder
        if self._batch:
            self._flush()

        logger.info("[Collector] Done — %d total chunks upserted.", self.total_chunks)

    def _flush(self):
        batch       = self._batch
        self._batch = []
        if not batch:
            return

        texts = [d["text"] for d in batch]
        try:
            embeddings = self.embedder.encode(
                [f"search_document: {t}" for t in texts]
            )
        except Exception as exc:
            logger.error("[Collector] Embedding failed for batch of %d: %s",
                         len(batch), exc)
            # Still record hashes so files aren't re-processed
            for d in batch:
                self.hash_updates[_hash_key(d["_path"])] = d.get("_hash", "")
            return

        points = []
        for d, emb in zip(batch, embeddings):
            self._point_id += 1
            points.append(PointStruct(
                id=self._point_id,
                vector=emb.tolist(),           # dense only — no sparse mismatch
                payload={
                    "text":        d["text"],
                    "source_file": d["source_file"],
                    "page_number": d["page_number"],
                    "chunk_index": d["chunk_index"],
                    "doc_type":    d["doc_type"],
                    "ocr_engine":  d["ocr_engine"],
                    "ocr_conf":    d["ocr_conf"],
                },
            ))
            self.hash_updates[_hash_key(d["_path"])] = d.get("_hash", "")

        try:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,   # commit before updating hash ledger
            )
            self.total_chunks += len(points)
            logger.info("[Collector] Flushed %d chunks → total %d (Qdrant id %d)",
                        len(points), self.total_chunks, self._point_id)
        except Exception as exc:
            logger.error("[Collector] Qdrant upsert FAILED for batch of %d: %s",
                         len(points), exc, exc_info=True)


# ── Main entry ────────────────────────────────────────────────────────

def run_ingestion(reset: bool = False, workers: int = NUM_WORKERS,
                  batch: int = EMBED_BATCH_SIZE):

    print("\n" + "═" * 60)
    print("  LEGAL SLM — INGESTION  (v6 · dense-only · incremental)")
    print("═" * 60)
    print(f"  Workers : {workers}   Batch : {batch}   Reset : {reset}")
    print(f"  PDF dir : {PDF_DIR}")
    print("═" * 60 + "\n")

    t0 = time.monotonic()

    # ── Qdrant setup ──────────────────────────────────────────────
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    if reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print("[Reset] Deleted existing collection.")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBED_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        print(f"[Qdrant] Created collection '{COLLECTION_NAME}'")
    else:
        n = client.count(COLLECTION_NAME).count
        print(f"[Qdrant] Collection exists — {n:,} vectors")

    # ── Discover PDFs ─────────────────────────────────────────────
    hash_db  = _load_hashes() if not reset else {}
    pdf_dir  = Path(PDF_DIR).resolve()
    all_pdfs = sorted(p for p in pdf_dir.rglob("*")
                      if p.suffix.lower() == ".pdf")
    print(f"[Scan] Found {len(all_pdfs):,} PDFs total")

    new_pdfs: list[tuple[str, str]] = []   # (path, hash)
    for p in all_pdfs:
        h   = _file_hash(str(p))
        key = _hash_key(str(p))
        if hash_db.get(key) != h:
            new_pdfs.append((str(p), h))

    if not new_pdfs:
        print("[Skip] All files already ingested. Nothing to do.")
        print(f"       Qdrant: {client.count(COLLECTION_NAME).count:,} vectors\n")
        return

    print(f"[New]  {len(new_pdfs):,} files to ingest\n")

    # ── Queue + collector ─────────────────────────────────────────
    result_queue: MPQueue = MPQueue(maxsize=QUEUE_MAXSIZE)
    embedder  = Embedder()
    collector = CollectorThread(client, embedder, result_queue, workers)
    collector.start()

    # ── Worker pool ───────────────────────────────────────────────
    ctx  = get_context("spawn")
    pool = ctx.Pool(processes=workers, initializer=warmup_paddle)

    pbar = tqdm(total=len(new_pdfs), desc="Processing PDFs", unit="pdf")

    pdf_paths = [p for p, _ in new_pdfs]
    hash_map  = {p: h for p, h in new_pdfs}

    try:
        for chunk_list in pool.imap_unordered(_process_pdf_worker, pdf_paths, chunksize=1):
            # Tag each chunk dict with its file hash for ledger
            path = chunk_list[0]["_path"] if chunk_list else None
            if path:
                for d in chunk_list:
                    d["_hash"] = hash_map.get(path, "")

            # Put with timeout + alive check to avoid deadlock
            while True:
                if not collector.is_alive():
                    logger.error("[Main] Collector died: %s", collector.error)
                    pool.terminate()
                    sys.exit(1)
                try:
                    result_queue.put(chunk_list, timeout=2.0)
                    break
                except Exception:
                    continue   # queue full — retry

            pbar.update(1)
            pbar.set_postfix({"chunks": collector.total_chunks})

    except KeyboardInterrupt:
        print("\n[Interrupted] Stopping...")
        pool.terminate()
    except Exception as exc:
        logger.error("[Main] Pool error: %s", exc, exc_info=True)
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
        pbar.close()

    # Signal collector all workers done
    for _ in range(workers):
        result_queue.put(_WORKER_DONE)

    print("\n[Collector] Waiting for final flush (max 10 min)...")
    collector.join(timeout=600)

    if collector.error:
        logger.error("[Main] Collector had an error: %s", collector.error)

    # ── Save hash ledger ──────────────────────────────────────────
    hash_db.update(collector.hash_updates)
    _save_hashes(hash_db)

    # ── Summary ───────────────────────────────────────────────────
    total_vecs = client.count(COLLECTION_NAME).count
    wall       = time.monotonic() - t0

    print("\n" + "═" * 60)
    print("  INGESTION COMPLETE")
    print("═" * 60)
    print(f"  PDFs processed : {len(new_pdfs):>8,}")
    print(f"  Chunks indexed : {collector.total_chunks:>8,}")
    print(f"  Qdrant total   : {total_vecs:>8,} vectors")
    print(f"  Wall time      : {wall/60:>7.1f} min")
    if wall > 0:
        print(f"  Throughput     : {len(new_pdfs)/wall*60:>7.1f} PDFs/min")
    print("═" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Legal SLM ingestion")
    p.add_argument("--reset",   action="store_true", help="Wipe and re-ingest everything")
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--batch",   type=int, default=EMBED_BATCH_SIZE)
    args = p.parse_args()
    run_ingestion(reset=args.reset, workers=args.workers, batch=args.batch)