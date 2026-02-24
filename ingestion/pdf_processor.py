"""
ingestion/pdf_processor.py  (v2 — PaddleOCR primary engine)

Per-page extraction with a 3-tier OCR cascade:

    DIGITAL      →  PyMuPDF direct text   (5ms,   confidence 1.0)
    SCANNED      →  PaddleOCR             (180ms, confidence 0.55–0.98)
                    └─ low conf / Deva    →  EasyOCR fallback
    HANDWRITTEN  →  PaddleOCR             (220ms, confidence 0.45–0.90)
                    └─ low conf           →  EasyOCR fallback
                    └─ Devanagari flag    →  EasyOCR (hi+en model)
    BLANK        →  skip

Why PaddleOCR as primary over TrOCR:
  - TrOCR-large: 2500ms/page → 145 hours for 208k pages (unacceptable)
  - PaddleOCR:    180ms/page →  10 hours for 208k pages (before parallelism)
  - With 4 parallel workers: ~2.5 hours for full 26k-doc corpus
  - Accuracy on degraded prints: PaddleOCR ≥ TrOCR for most Indian legal docs
  - TrOCR is still the best for isolated English cursive handwriting —
    it is preserved as an optional tertiary pass (see TERTIARY_TROCR flag)

Cascade decision tree
─────────────────────
                   classify_page()
                        │
            ┌───────────┼──────────────┬────────┐
         DIGITAL    SCANNED      HANDWRITTEN  BLANK
            │           │              │        │
         pymupdf    paddle_ocr    paddle_ocr   skip
            │        conf≥0.55?    conf≥0.45?
            |       /      \\       /      \\
         done    done   easyocr  done  easyocr
                              lang=deva?
                                  │
                            easyocr(hi+en)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

from ingestion.document_classifier import classify_page, PageType
from ingestion.ocr_engine import preprocess_for_ocr, EasyOCREngine
from ingestion.paddle_ocr_engine import (
    extract_with_paddle,
    OCRResult,
    PADDLE_CONFIDENCE_FLOOR,
    PADDLE_AVAILABLE,
)
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# ── Feature flags ────────────────────────────────────────────────────
# Set TERTIARY_TROCR=True to enable TrOCR as a final pass for pages
# where both PaddleOCR AND EasyOCR confidence is below HANDWRITTEN_FLOOR.
# Trades speed for maximum accuracy on the worst-quality handwritten pages.
TERTIARY_TROCR      = False
HANDWRITTEN_FLOOR   = 0.45   # Minimum acceptable confidence for HW pages
SCANNED_FLOOR       = 0.55   # Minimum acceptable confidence for scanned pages
DIGITAL_MIN_CHARS   = 15     # Below this, digital text is treated as failed


# ────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """One chunk of text with complete provenance metadata."""
    text:            str
    source_file:     str
    page_number:     int       # 1-indexed
    chunk_index:     int       # position within the page
    doc_type:        str       # FIR | Panchnama | Judgment | Unknown
    page_type:       str       # digital | scanned | handwritten | blank
    ocr_engine:      str       # pymupdf | paddle | easyocr | trocr
    ocr_confidence:  float     # 0.0 – 1.0


@dataclass
class PageExtractionLog:
    """Full per-page audit record.  Written to the ingestion log file."""
    filename:            str
    page_number:         int
    page_type:           str
    primary_engine:      str
    primary_confidence:  float
    fallback_used:       bool
    fallback_engine:     str | None
    final_confidence:    float
    text_length:         int
    elapsed_ms:          float
    needs_review:        bool    # True when final_confidence < floor
    warnings:            list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Module-level engine singletons  (one per worker process)
# ────────────────────────────────────────────────────────────────────

# EasyOCR is lazy-loaded the first time it is needed for a fallback.
# This avoids consuming ~1.5 GB of RAM when every page is clean.
_easyocr_engine: EasyOCREngine | None = None


def _get_easyocr() -> EasyOCREngine:
    global _easyocr_engine
    if _easyocr_engine is None:
        logger.info("[EasyOCR] Loading multilingual model (en+hi)...")
        _easyocr_engine = EasyOCREngine()
    return _easyocr_engine


# TrOCR tertiary engine — only loaded when TERTIARY_TROCR=True
_trocr_engine = None


def _get_trocr():
    global _trocr_engine
    if _trocr_engine is None:
        from ingestion.ocr_engine import TrOCREngine
        logger.info("[TrOCR] Loading large handwriting model (one-time ~10s)...")
        _trocr_engine = TrOCREngine()
    return _trocr_engine


# ────────────────────────────────────────────────────────────────────
# Document-type heuristic
# ────────────────────────────────────────────────────────────────────

def detect_doc_type(filename: str) -> str:
    fname = filename.upper()
    if "FIR"    in fname:               return "FIR"
    if "PANCH"  in fname:               return "Panchnama"
    if "JUDG"   in fname or "ORDER" in fname: return "Judgment"
    return "Unknown"


# ────────────────────────────────────────────────────────────────────
# Chunking
# ────────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Word-boundary chunker with overlap.
    Overlap preserves context for legal clauses that span chunk boundaries
    (e.g., an IPC section that starts at the end of one chunk and whose
    punishment appears at the start of the next).
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        # Slide forward by (CHUNK_SIZE - CHUNK_OVERLAP) words
        start += max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks


# ────────────────────────────────────────────────────────────────────
# Per-page extraction with 3-tier cascade
# ────────────────────────────────────────────────────────────────────

def _extract_digital(page: fitz.Page) -> tuple[str, PageExtractionLog | None]:
    """Fast path: pull selectable text directly from the PDF font layer."""
    # sort=True: honour reading order (left→right, top→bottom)
    text = page.get_text("text", sort=True).strip()
    return text, None   # Log built by caller


def _run_ocr_cascade(
    page_img,           # PIL Image (pre-processed)
    page_type: PageType,
    filename: str,
    page_num: int,
) -> OCRResult:
    """
    Execute the PaddleOCR → EasyOCR → (TrOCR) cascade.

    Returns the best OCRResult found, with `engine_used` set to
    whichever engine produced it.
    """
    confidence_floor = (
        HANDWRITTEN_FLOOR
        if page_type == PageType.HANDWRITTEN
        else SCANNED_FLOOR
    )

    # ── Tier 1: PaddleOCR ──────────────────────────────────────────
    if PADDLE_AVAILABLE:
        paddle_result = extract_with_paddle(
            page_img,
            check_devanagari=(page_type == PageType.HANDWRITTEN),
        )
        logger.debug(
            "[%s p%d] Paddle conf=%.3f lang=%s elapsed=%.0fms",
            filename, page_num,
            paddle_result.confidence,
            paddle_result.language_detected,
            paddle_result.elapsed_ms,
        )
    else:
        # Graceful degradation if PaddleOCR isn't installed
        logger.warning(
            "[%s p%d] PaddleOCR not available — falling back to EasyOCR directly",
            filename, page_num,
        )
        paddle_result = OCRResult(
            text="", confidence=0.0,
            engine_used="paddle_unavailable",
            language_detected="unknown",
        )

    # ── Early-exit: Paddle is good enough ─────────────────────────
    needs_fallback = (
        paddle_result.confidence < confidence_floor
        or paddle_result.language_detected in ("devanagari", "unknown")
    )

    if not needs_fallback:
        return paddle_result

    # ── Tier 2: EasyOCR fallback ───────────────────────────────────
    logger.debug(
        "[%s p%d] Paddle below threshold (%.3f < %.3f) or Devanagari "
        "detected — running EasyOCR",
        filename, page_num, paddle_result.confidence, confidence_floor,
    )

    easyocr_engine  = _get_easyocr()
    easyocr_result  = easyocr_engine.extract(page_img)

    best_result = (
        easyocr_result
        if easyocr_result.confidence > paddle_result.confidence
        else paddle_result
    )

    # ── Tier 3: TrOCR (optional, handwritten English only) ─────────
    if (
        TERTIARY_TROCR
        and page_type == PageType.HANDWRITTEN
        and best_result.confidence < confidence_floor
        and best_result.language_detected not in ("devanagari",)
    ):
        logger.debug(
            "[%s p%d] Both Paddle and EasyOCR low (%.3f) — trying TrOCR",
            filename, page_num, best_result.confidence,
        )
        trocr_result = _get_trocr().extract(page_img)
        if trocr_result.confidence > best_result.confidence:
            best_result = trocr_result

    return best_result


def extract_page_text(
    page: fitz.Page,
    page_type: PageType,
    page_img,           # PIL Image or None (None for DIGITAL pages)
    filename: str,
    page_num: int,
) -> tuple[str, PageExtractionLog]:
    """
    Extract text from one page. Dispatches to the correct tier and
    returns (text, log).

    This is the single function called by process_pdf() — the rest of
    the module is implementation detail.
    """
    import time

    t_start    = time.monotonic()
    warnings   = []
    fallback   = False
    fallback_e = None

    # ── BLANK ────────────────────────────────────────────────────────
    if page_type == PageType.BLANK:
        log = PageExtractionLog(
            filename=filename, page_number=page_num,
            page_type="blank",
            primary_engine="none", primary_confidence=0.0,
            fallback_used=False, fallback_engine=None,
            final_confidence=0.0, text_length=0,
            elapsed_ms=(time.monotonic() - t_start) * 1000,
            needs_review=False,
            warnings=["Page classified as blank — skipped"],
        )
        return "", log

    # ── DIGITAL ──────────────────────────────────────────────────────
    if page_type == PageType.DIGITAL:
        text = page.get_text("text", sort=True).strip()

        if len(text) < DIGITAL_MIN_CHARS:
            # PDF claims to have a text layer but it's almost empty
            # (common with poorly-generated PDFs where text is invisible/white).
            # Fall through to OCR.
            warnings.append(
                f"Digital text layer present but only {len(text)} chars — "
                "running OCR as safety net"
            )
            preprocessed = preprocess_for_ocr(page_img or _render_page(page), "scanned")
            result = _run_ocr_cascade(preprocessed, PageType.SCANNED, filename, page_num)
            text   = result.text
            engine = result.engine_used
            conf   = result.confidence
            fallback = True
        else:
            engine   = "pymupdf"
            conf     = 1.0

        log = PageExtractionLog(
            filename=filename, page_number=page_num,
            page_type="digital",
            primary_engine=engine, primary_confidence=conf,
            fallback_used=fallback, fallback_engine=engine if fallback else None,
            final_confidence=conf, text_length=len(text),
            elapsed_ms=(time.monotonic() - t_start) * 1000,
            needs_review=False,
            warnings=warnings,
        )
        return text, log

    # ── SCANNED or HANDWRITTEN: OCR cascade ──────────────────────────
    preprocessed = preprocess_for_ocr(page_img, page_type.value)
    result       = _run_ocr_cascade(preprocessed, page_type, filename, page_num)

    primary_engine = result.engine_used
    primary_conf   = result.confidence
    fallback_used  = result.engine_used not in ("paddle", "paddle_unavailable")
    fallback_e     = result.engine_used if fallback_used else None

    # Determine confidence floor for this page type
    floor = (
        HANDWRITTEN_FLOOR
        if page_type == PageType.HANDWRITTEN
        else SCANNED_FLOOR
    )

    needs_review = result.confidence < floor
    if needs_review:
        warnings.append(
            f"Final OCR confidence {result.confidence:.3f} below "
            f"threshold {floor:.2f}. Flagged for manual review."
        )

    log = PageExtractionLog(
        filename=filename, page_number=page_num,
        page_type=page_type.value,
        primary_engine=primary_engine,
        primary_confidence=primary_conf,
        fallback_used=fallback_used,
        fallback_engine=fallback_e,
        final_confidence=result.confidence,
        text_length=len(result.text),
        elapsed_ms=(time.monotonic() - t_start) * 1000,
        needs_review=needs_review,
        warnings=warnings,
    )
    return result.text, log


def _render_page(page: fitz.Page, dpi: int = 300):
    """Render a fitz.Page to PIL Image (used as safety fallback)."""
    from PIL import Image
    import io
    zoom = dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")


# ────────────────────────────────────────────────────────────────────
# Main generator
# ────────────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    verbose_logs: bool = False,
) -> Generator[tuple[TextChunk, PageExtractionLog], None, None]:

    filename = Path(pdf_path).name
    doc_type = detect_doc_type(filename)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("[%s] Cannot open PDF: %s", filename, exc)
        return

    pdf_has_chunks = False  # ⭐ guarantee at least one chunk per PDF

    for page_num, page in enumerate(doc, start=1):

        # ── 1. Classify page ─────────────────────────────────────
        page_type, page_img = classify_page(page)

        # ⭐ FIX — prevent misclassification loss
        if page_type == PageType.SCANNED and page.get_text("text").strip():
            page_type = PageType.DIGITAL

        # ── 2. Extract text ──────────────────────────────────────
        text, log = extract_page_text(page, page_type, page_img, filename, page_num)

        if verbose_logs:
            flag = "⚠ REVIEW" if log.needs_review else "✓"
            print(
                f"  {flag} {filename} p{page_num:04d} "
                f"[{page_type.value:>12s}] "
                f"engine={log.primary_engine:<8s} "
                f"conf={log.final_confidence:.2f} "
                f"({log.elapsed_ms:.0f}ms)"
            )

        # ⭐ FIX — hard fallback to prevent data loss
        if not text or len(text.strip()) < 20:
            try:
                page_img = _render_page(page)
                preprocessed = preprocess_for_ocr(page_img, "scanned")
                result = _run_ocr_cascade(preprocessed, PageType.SCANNED, filename, page_num)
                text = result.text
            except Exception:
                text = ""

        # ⭐ FINAL GUARANTEE
        if not text or len(text.strip()) < 10:
            logger.warning(f"[{filename} p{page_num}] extraction failed — storing placeholder")
            text = "[UNREADABLE PAGE]"

        # ── 3. Chunk and yield ───────────────────────────────────
        chunks = chunk_text(text)

        if not chunks:
            chunks = ["[EMPTY PAGE]"]

        for idx, chunk in enumerate(chunks):
            pdf_has_chunks = True
            tc = TextChunk(
                text=chunk,
                source_file=filename,
                page_number=page_num,
                chunk_index=idx,
                doc_type=doc_type,
                page_type=page_type.value,
                ocr_engine=log.primary_engine,
                ocr_confidence=log.final_confidence,
            )
            yield tc, log

    doc.close()

    # ⭐ CRITICAL FIX — PDF-LEVEL GUARANTEE
    if not pdf_has_chunks:
        logger.warning(f"[{filename}] No extractable text — inserting placeholder chunk")

        yield TextChunk(
            text="[EMPTY DOCUMENT]",
            source_file=filename,
            page_number=1,
            chunk_index=0,
            doc_type=doc_type,
            page_type="unknown",
            ocr_engine="fallback",
            ocr_confidence=0.0,
        ), PageExtractionLog(
            filename=filename,
            page_number=1,
            page_type="unknown",
            primary_engine="fallback",
            primary_confidence=0.0,
            fallback_used=True,
            fallback_engine="fallback",
            final_confidence=0.0,
            text_length=0,
            elapsed_ms=0,
            needs_review=True,
            warnings=["No extractable text found"],
        )