"""
ingestion/ocr_engine.py

OCR engine implementations:
  - TesseractEngine  : for clean printed/scanned pages
  - TrOCREngine      : for English handwritten pages (optional tertiary)
  - EasyOCREngine    : for multilingual (English + Hindi) fallback
  - preprocess_for_ocr: shared image pre-processing pipeline

PaddleOCR (primary engine) lives in paddle_ocr_engine.py.
This module provides the fallback engines and shared preprocessing.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

# ── Lazy imports (heavy models loaded only when needed) ──────────────
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    _DEVICE = "cpu"

try:
    import easyocr as _easyocr_lib
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────
# Shared result type
# ────────────────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    text: str
    confidence: float        # 0.0 – 1.0
    engine_used: str
    language_detected: str
    elapsed_ms: float = 0.0


# ────────────────────────────────────────────────────────────────────
# Image pre-processing  (shared by all engines)
# ────────────────────────────────────────────────────────────────────

def preprocess_for_ocr(img: Image.Image, page_type: str) -> Image.Image:
    """
    Pre-process a PIL image before passing to any OCR engine.

    Handles the main degradation modes in 1960s Indian legal documents:
      - Faded ink (low contrast)
      - Yellowed paper (colour noise)
      - Slight blur from microfilm/flatbed scanning
      - Skew (rotation during digitisation)
      - Bleed-through from thin paper

    Args:
        img:       Input PIL image (any mode).
        page_type: "scanned" | "handwritten" — affects dilation step.

    Returns:
        Pre-processed grayscale PIL Image ready for OCR.
    """
    # Convert to grayscale
    if img.mode != "L":
        img = img.convert("L")

    # Step 1: Contrast enhancement — critical for faded ink
    img = ImageEnhance.Contrast(img).enhance(2.5)

    # Step 2: Sharpening — helps with slightly out-of-focus scans
    img = img.filter(ImageFilter.SHARPEN)

    # Step 3: Adaptive binarisation via local mean subtraction
    arr        = np.array(img, dtype=np.float32)
    try:
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(arr, size=51)
        binary     = ((arr - local_mean) > -10).astype(np.uint8) * 255
        img        = Image.fromarray(binary)
    except ImportError:
        # scipy not available — fall back to simple global threshold
        threshold = int(arr.mean())
        img       = img.point(lambda p: 255 if p > threshold else 0)

    # Step 4: Deskew
    img = _deskew(img)

    # Step 5: For handwritten pages — dilate strokes slightly so thin
    # cursive strokes don't disappear after binarisation
    if page_type == "handwritten":
        img = img.filter(ImageFilter.MinFilter(3))

    return img


def _deskew(img: Image.Image) -> Image.Image:
    """Correct small rotation angles (up to ±10°) using Hough lines."""
    try:
        import cv2
        arr   = np.array(img)
        edges = cv2.Canny(arr, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return img
        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) < 10:
                angles.append(angle)
        if not angles:
            return img
        median_angle = float(np.median(angles))
        if abs(median_angle) > 0.5:
            img = img.rotate(
                -median_angle,
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=255,
            )
        return img
    except Exception:
        return img


# ────────────────────────────────────────────────────────────────────
# Tesseract engine
# ────────────────────────────────────────────────────────────────────

class TesseractEngine:
    """
    Tesseract 5 LSTM engine.
    Best for: clean printed scans, typed carbon-copy documents.
    Supports English + Hindi + Marathi via language packs.
    Install language packs: brew install tesseract-lang
    """

    _CONFIG = (
        "--oem 1 "           # LSTM engine
        "--psm 6 "           # Uniform block of text
        "-c preserve_interword_spaces=1"
    )

    _LANG_MAP = {
        "english":      "eng",
        "hindi":        "eng+hin",
        "marathi":      "eng+mar",
        "multilingual": "eng+hin+mar",
    }

    def extract(
        self,
        img: Image.Image,
        language: str = "multilingual",
    ) -> OCRResult:
        if not TESSERACT_AVAILABLE:
            return OCRResult(
                text="", confidence=0.0,
                engine_used="tesseract_unavailable",
                language_detected=language,
            )

        lang_code = self._LANG_MAP.get(language, "eng+hin+mar")

        try:
            data = pytesseract.image_to_data(
                img,
                lang=lang_code,
                config=self._CONFIG,
                output_type=pytesseract.Output.DICT,
            )
            words, confs = [], []
            for word, conf in zip(data["text"], data["conf"]):
                c = int(conf)
                if c > 30 and str(word).strip():
                    words.append(word)
                    confs.append(c)

            text       = " ".join(words)
            avg_conf   = (sum(confs) / len(confs) / 100.0) if confs else 0.0

            return OCRResult(
                text=text,
                confidence=avg_conf,
                engine_used="tesseract",
                language_detected=language,
            )
        except Exception as exc:
            logger.warning("[Tesseract] Failed: %s", exc)
            return OCRResult(
                text="", confidence=0.0,
                engine_used="tesseract",
                language_detected=language,
            )


# ────────────────────────────────────────────────────────────────────
# TrOCR engine  (optional tertiary — English handwriting only)
# ────────────────────────────────────────────────────────────────────

class TrOCREngine:
    """
    Microsoft TrOCR large handwritten model.
    Only loaded when TERTIARY_TROCR=True in pdf_processor.py.
    ~2.5s/page but highest accuracy for cursive English handwriting.
    NOT suitable for Devanagari — use EasyOCR for that.
    """

    MODEL_ID = "microsoft/trocr-large-handwritten"

    def __init__(self):
        self._loaded    = False
        self.processor  = None
        self.model      = None

    def _lazy_load(self):
        if self._loaded:
            return
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for TrOCR")
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        logger.info("[TrOCR] Loading %s (one-time ~10s)...", self.MODEL_ID)
        self.processor = TrOCRProcessor.from_pretrained(self.MODEL_ID)
        self.model     = VisionEncoderDecoderModel.from_pretrained(
            self.MODEL_ID
        ).to(_DEVICE)
        self.model.eval()
        self._loaded = True

    def extract(self, img: Image.Image) -> OCRResult:
        import time
        t0 = time.monotonic()
        try:
            self._lazy_load()
        except Exception as exc:
            logger.warning("[TrOCR] Load failed: %s", exc)
            return OCRResult(
                text="", confidence=0.0,
                engine_used="trocr", language_detected="english",
            )

        lines     = self._segment_lines(img)
        all_text  = []
        for line_img in lines:
            text = self._process_line(line_img)
            if text.strip():
                all_text.append(text)

        full_text  = "\n".join(all_text)
        confidence = min(1.0, len(full_text.split()) / 50.0) if full_text else 0.0
        elapsed_ms = (time.monotonic() - t0) * 1000

        return OCRResult(
            text=full_text,
            confidence=confidence,
            engine_used="trocr",
            language_detected="english",
            elapsed_ms=elapsed_ms,
        )

    def _process_line(self, line_img: Image.Image) -> str:
        import torch
        if line_img.mode != "RGB":
            line_img = line_img.convert("RGB")
        line_img = line_img.resize(
            (384, max(32, line_img.height)), Image.LANCZOS
        )
        pixel_values = self.processor(
            images=line_img, return_tensors="pt"
        ).pixel_values.to(_DEVICE)
        with torch.no_grad():
            ids = self.model.generate(
                pixel_values, max_new_tokens=128,
                num_beams=4, early_stopping=True,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def _segment_lines(
        self, img: Image.Image, min_height: int = 20
    ) -> list[Image.Image]:
        arr    = np.array(img.convert("L"))
        binary = (arr < 128).astype(np.float32)
        h_proj = binary.sum(axis=1)
        in_line = h_proj > (h_proj.max() * 0.05)
        lines, start = [], None
        for i, is_text in enumerate(in_line):
            if is_text and start is None:
                start = i
            elif not is_text and start is not None:
                if (i - start) >= min_height:
                    top    = max(0, start - 4)
                    bottom = min(img.height, i + 4)
                    lines.append(img.crop((0, top, img.width, bottom)))
                start = None
        if start is not None:
            lines.append(img.crop((0, start, img.width, img.height)))
        return lines if lines else [img]


# ────────────────────────────────────────────────────────────────────
# EasyOCR engine  (primary fallback — multilingual)
# ────────────────────────────────────────────────────────────────────

class EasyOCREngine:
    """
    EasyOCR with English + Hindi support.
    Used as fallback when PaddleOCR confidence is low, or when
    Devanagari script is detected (TrOCR cannot handle it).

    Slower than PaddleOCR (~1.8s/page) but more robust on degraded
    multi-script documents.
    """

    def __init__(self):
        self._reader = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        if not EASYOCR_AVAILABLE:
            raise RuntimeError(
                "easyocr not installed. Run: pip install easyocr"
            )
        logger.info("[EasyOCR] Loading en+hi model (one-time ~8s)...")
        use_gpu = TORCH_AVAILABLE and __import__("torch").backends.mps.is_available()
        self._reader = _easyocr_lib.Reader(["en", "hi"], gpu=use_gpu)
        self._loaded = True

    def extract(self, img: Image.Image) -> OCRResult:
        import time
        t0 = time.monotonic()
        try:
            self._lazy_load()
        except Exception as exc:
            logger.warning("[EasyOCR] Load failed: %s", exc)
            return OCRResult(
                text="", confidence=0.0,
                engine_used="easyocr", language_detected="unknown",
            )

        arr = np.array(img)
        try:
            results = self._reader.readtext(
                arr,
                detail=1,
                paragraph=True,
                contrast_ths=0.1,
                adjust_contrast=0.5,
            )
        except Exception as exc:
            logger.warning("[EasyOCR] readtext failed: %s", exc)
            return OCRResult(
                text="", confidence=0.0,
                engine_used="easyocr", language_detected="unknown",
            )

        if not results:
            return OCRResult(
                text="", confidence=0.0,
                engine_used="easyocr", language_detected="unknown",
            )

        texts = [r[1] for r in results]
        confs = [r[2] for r in results]
        elapsed_ms = (time.monotonic() - t0) * 1000

        return OCRResult(
            text="\n".join(texts),
            confidence=float(np.mean(confs)) if confs else 0.0,
            engine_used="easyocr",
            language_detected="multilingual",
            elapsed_ms=elapsed_ms,
        )