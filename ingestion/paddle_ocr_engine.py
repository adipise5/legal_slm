"""
ingestion/paddle_ocr_engine.py
Updated for PaddleOCR v3.4.0 + Python 3.14 Stability.
"""
import logging
import os
import time
import sys
import types
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image

# ── MONKEYPATCH FOR PADDLEOCR v3 + PYTHON 3.14 ──
try:
    import langchain
except ImportError:
    from importlib.machinery import ModuleSpec
    m_langchain = types.ModuleType("langchain")
    m_langchain.__path__ = []
    sys.modules["langchain"] = m_langchain
    doc_mod = types.ModuleType("langchain.docstore")
    sys.modules["langchain.docstore"] = doc_mod
    docdoc = types.ModuleType("langchain.docstore.document")
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    docdoc.Document = Document
    sys.modules["langchain.docstore.document"] = docdoc

logger = logging.getLogger(__name__)

# ── Constants required by pdf_processor.py ──
PADDLE_CONFIDENCE_FLOOR = 0.55

PADDLE_AVAILABLE = False
try:
    import paddleocr
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    logger.warning(f"[PaddleOCR] Import failed: {e}")

@dataclass
class OCRResult:
    text: str
    confidence: float
    engine_used: str
    language_detected: str
    elapsed_ms: float = 0.0

class _PaddleOCRSingleton:
    _instance = None
    _ocr = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_ocr(self):
        if self._ocr is not None:
            return self._ocr
        # V3.4.0 FIX: Removed legacy args use_gpu/show_log to avoid ValueErrors
        self._ocr = PaddleOCR(lang="en") 
        return self._ocr

_singleton = _PaddleOCRSingleton()

def extract_with_paddle(img: Image.Image, check_devanagari: bool = False) -> OCRResult:
    if not PADDLE_AVAILABLE:
        return OCRResult("", 0.0, "unavailable", "unknown")
    t0 = time.monotonic()
    ocr = _singleton.get_ocr()
    arr = np.array(img.convert("RGB"))
    
    results = ocr.ocr(arr, cls=True)
    texts, confs = [], []
    if results and results[0]:
        for line in results[0]:
            if line and len(line) >= 2:
                texts.append(str(line[1][0]).strip())
                confs.append(float(line[1][1]))
    
    full_text = "\n".join(texts)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return OCRResult(full_text, avg_conf, "paddle", "english", (time.monotonic() - t0) * 1000)

def warmup_paddle():
    if PADDLE_AVAILABLE:
        _singleton.get_ocr()