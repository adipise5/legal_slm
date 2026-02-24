"""
ingestion/document_classifier.py

Per-page document type classifier.
Determines whether each PDF page contains:
  - DIGITAL     : selectable text layer (PyMuPDF direct extraction)
  - SCANNED     : printed text scanned as image (Tesseract / PaddleOCR)
  - HANDWRITTEN : handwritten content (PaddleOCR / TrOCR / EasyOCR)
  - BLANK       : empty or fully degraded page (skip)

Classification is per-page because a single 1960s legal PDF can have
mixed content — e.g. a typed FIR with handwritten additions in margins.
"""

from __future__ import annotations

import io
from enum import Enum

import numpy as np
from PIL import Image, ImageStat
import fitz

# ⭐ Reduced threshold — critical fix
DIGITAL_MIN_CHARS = 15
HANDWRITING_THRESHOLD = 0.45


class PageType(Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    HANDWRITTEN = "handwritten"
    BLANK = "blank"


def page_to_pil(page: fitz.Page, dpi: int = 300) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")


def has_digital_text(page: fitz.Page) -> bool:
    """
    Improved digital detection:
    ✔ lower threshold
    ✔ ignore whitespace
    ✔ ignore isolated characters
    """
    text = page.get_text("text")
    clean = " ".join(text.split())

    # ⭐ FIX 1 — low threshold
    if len(clean) >= DIGITAL_MIN_CHARS:
        return True

    # ⭐ FIX 2 — fallback glyph check
    words = page.get_text("words")
    if len(words) > 5:
        return True

    return False


def estimate_handwriting_probability(img: Image.Image) -> float:
    arr = np.array(img, dtype=np.float32)

    threshold = arr.mean()
    binary = (arr < threshold).astype(np.float32)

    ink_ratio = binary.mean()
    if ink_ratio < 0.01 or ink_ratio > 0.8:
        return 0.0

    h_proj = binary.sum(axis=1)
    if h_proj.max() == 0:
        return 0.0
    h_norm = h_proj / h_proj.max()
    nonzero = h_norm[h_norm > 0.05]
    cv_h = (nonzero.std() / (nonzero.mean() + 1e-9)) if len(nonzero) > 5 else 0.0

    transitions = np.diff((h_norm > 0.1).astype(int))
    transition_ratio = np.abs(transitions).sum() / (len(h_norm) + 1e-9)

    v_proj = binary.sum(axis=0)
    v_norm = v_proj / (v_proj.max() + 1e-9)
    v_nonzero = v_norm[v_norm > 0.05]
    cv_v = (v_nonzero.std() / (v_nonzero.mean() + 1e-9)) if len(v_nonzero) > 5 else 0.0

    score = (cv_h * 0.5) + (transition_ratio * 2.0) + (cv_v * 0.3)
    return float(min(1.0, score / 2.5))


def classify_page(page: fitz.Page) -> tuple[PageType, Image.Image | None]:

    # ⭐ DIGITAL fast path
    if has_digital_text(page):
        return PageType.DIGITAL, None

    img = page_to_pil(page, dpi=300)

    stat = ImageStat.Stat(img)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]

    if brightness > 245 or contrast < 8:
        return PageType.BLANK, img

    hw_prob = estimate_handwriting_probability(img)

    if hw_prob > HANDWRITING_THRESHOLD:
        return PageType.HANDWRITTEN, img
    else:
        return PageType.SCANNED, img