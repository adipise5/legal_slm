"""
diagnose.py — run this BEFORE ingestion to find extraction problems.

Usage:
    python diagnose.py                  # tests 20 random PDFs
    python diagnose.py --n 50           # tests 50 PDFs
    python diagnose.py --pdf path.pdf   # tests one specific PDF
    python diagnose.py --fix            # also shows what fix is needed
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import traceback
from pathlib import Path
from collections import Counter

# ── must be run from project root ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


def test_one_pdf(pdf_path: str, verbose: bool = False) -> dict:
    """
    Run the full extraction pipeline on one PDF and return a report.
    """
    import fitz
    from ingestion.document_classifier import classify_page, PageType
    from ingestion.pdf_processor import process_pdf, chunk_text, detect_doc_type
    from ingestion.paddle_ocr_engine import PADDLE_AVAILABLE

    result = {
        "path":        pdf_path,
        "name":        Path(pdf_path).name,
        "pages":       0,
        "chunks":      0,
        "page_types":  Counter(),
        "ocr_engines": Counter(),
        "errors":      [],
        "sample_text": "",
        "empty_pages": 0,
        "placeholder_chunks": 0,
    }

    try:
        doc = fitz.open(pdf_path)
        result["pages"] = len(doc)
        doc.close()
    except Exception as e:
        result["errors"].append(f"Cannot open PDF: {e}")
        return result

    try:
        chunks_seen = []
        for chunk, log in process_pdf(pdf_path, verbose_logs=verbose):
            result["chunks"] += 1
            result["page_types"][log.page_type] += 1
            result["ocr_engines"][log.primary_engine] += 1
            if chunk.text in ("[EMPTY PAGE]", "[UNREADABLE PAGE]", "[EMPTY DOCUMENT]", "[EXTRACTION FAILED]"):
                result["placeholder_chunks"] += 1
            else:
                chunks_seen.append(chunk.text[:100])

        if chunks_seen and not result["sample_text"]:
            result["sample_text"] = chunks_seen[0]

        result["empty_pages"] = result["placeholder_chunks"]

    except Exception as e:
        result["errors"].append(traceback.format_exc())

    return result


def classify_only(pdf_path: str) -> dict:
    """Just run page classification without OCR — fast diagnostic."""
    import fitz
    from ingestion.document_classifier import classify_page, PageType

    counts = Counter()
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pt, _ = classify_page(page)
            counts[pt.value] += 1
        doc.close()
    except Exception as e:
        return {"error": str(e)}
    return dict(counts)


def run_diagnosis(n: int = 20, specific_pdf: str = None, show_fix: bool = False):
    from config import PDF_DIR
    from ingestion.paddle_ocr_engine import PADDLE_AVAILABLE

    print("\n" + "═" * 65)
    print("  LEGAL SLM — EXTRACTION DIAGNOSTICS")
    print("═" * 65)
    print(f"  PaddleOCR available : {PADDLE_AVAILABLE}")

    # Check config
    try:
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        print(f"  CHUNK_SIZE          : {CHUNK_SIZE}")
        print(f"  CHUNK_OVERLAP       : {CHUNK_OVERLAP}")
    except Exception as e:
        print(f"  config import error : {e}")

    print("═" * 65 + "\n")

    # ── Select PDFs to test ───────────────────────────────────────
    if specific_pdf:
        pdfs = [specific_pdf]
    else:
        pdf_dir = Path(PDF_DIR)
        all_pdfs = sorted(pdf_dir.rglob("*.pdf"))
        if not all_pdfs:
            print(f"❌ No PDFs found in {PDF_DIR}")
            return
        pdfs = random.sample(all_pdfs, min(n, len(all_pdfs)))
        pdfs = [str(p) for p in pdfs]

    print(f"Testing {len(pdfs)} PDFs...\n")

    # ── Run tests ──────────────────────────────────────────────────
    total_pages  = 0
    total_chunks = 0
    total_placeholder = 0
    all_page_types  = Counter()
    all_ocr_engines = Counter()
    zero_chunk_pdfs = []
    error_pdfs      = []

    for i, pdf in enumerate(pdfs, 1):
        r = test_one_pdf(pdf)

        status = "✓"
        if r["errors"]:
            status = "✗"
            error_pdfs.append(r)
        elif r["chunks"] == 0:
            status = "⚠ 0 CHUNKS"
            zero_chunk_pdfs.append(r)
        elif r["placeholder_chunks"] == r["chunks"]:
            status = "⚠ ALL PLACEHOLDER"
            zero_chunk_pdfs.append(r)

        total_pages      += r["pages"]
        total_chunks     += r["chunks"]
        total_placeholder+= r["placeholder_chunks"]
        all_page_types   += r["page_types"]
        all_ocr_engines  += r["ocr_engines"]

        print(f"  [{i:3d}/{len(pdfs)}] {status:20s} {r['name'][:40]:40s} "
              f"pages={r['pages']:3d} chunks={r['chunks']:4d} "
              f"placeholder={r['placeholder_chunks']:3d}")
        if r["sample_text"]:
            print(f"            sample: {repr(r['sample_text'][:80])}")
        if r["errors"]:
            print(f"            ERROR: {r['errors'][0][:120]}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  SUMMARY")
    print("─" * 65)
    print(f"  PDFs tested      : {len(pdfs)}")
    print(f"  Total pages      : {total_pages}")
    print(f"  Total chunks     : {total_chunks}")
    print(f"  Avg chunks/PDF   : {total_chunks/max(1,len(pdfs)):.1f}")
    print(f"  Avg chunks/page  : {total_chunks/max(1,total_pages):.2f}")
    print(f"  Placeholder chunks: {total_placeholder} / {total_chunks}")
    print(f"  PDFs with 0 chunks: {len(zero_chunk_pdfs)}")
    print(f"  PDFs with errors  : {len(error_pdfs)}")
    print()
    print("  Page type breakdown:")
    for pt, count in all_page_types.most_common():
        print(f"    {pt:15s} : {count}")
    print()
    print("  OCR engine breakdown:")
    for eng, count in all_ocr_engines.most_common():
        print(f"    {eng:15s} : {count}")

    # ── Diagnose root cause ────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  ROOT CAUSE ANALYSIS")
    print("─" * 65)

    avg_cpp = total_chunks / max(1, total_pages)

    if total_chunks == 0:
        print("  ❌ CRITICAL: Zero chunks produced from any PDF.")
        print("     → process_pdf() is raising an exception or returning nothing.")
        print("     → Run: python diagnose.py --pdf data/legal_pdfs/YOURFILE.pdf")
        print("       to see the full traceback for a specific file.")

    elif avg_cpp < 0.5:
        print("  ⚠ Very low chunks per page — most pages returning placeholders.")
        digital = all_page_types.get("digital", 0)
        scanned = all_page_types.get("scanned", 0)
        blank   = all_page_types.get("blank", 0)
        total_pt = sum(all_page_types.values()) or 1

        if blank / total_pt > 0.5:
            print("  → 50%+ pages classified BLANK.")
            print("    CAUSE: contrast/brightness thresholds too aggressive.")
            print("    FIX:   In document_classifier.py change:")
            print("           brightness > 245  →  brightness > 252")
            print("           contrast < 8      →  contrast < 3")

        elif digital / total_pt > 0.7 and total_chunks < total_pages:
            print("  → Most pages are DIGITAL but chunks still low.")
            print("    CAUSE: CHUNK_SIZE too large or text very short per page.")
            print("    FIX:   In config.py: CHUNK_SIZE = 100, CHUNK_OVERLAP = 20")

        elif scanned / total_pt > 0.5:
            print("  → Most pages are SCANNED but OCR producing little text.")
            if not PADDLE_AVAILABLE:
                print("    CAUSE: PaddleOCR not available — EasyOCR is slow fallback.")
            else:
                print("    CAUSE: PaddleOCR confidence below threshold → empty results.")
                print("    FIX:   In pdf_processor.py: SCANNED_FLOOR = 0.3")

    elif avg_cpp >= 1.0:
        print(f"  ✅ Extraction looks healthy ({avg_cpp:.1f} chunks/page).")
        print("     If total ingested chunks are low, check:")
        print("     1. Are all 2000 PDFs in data/legal_pdfs/ ?")
        print("     2. Is the hash ledger stale? Run with --reset")

    # ── Show zero-chunk PDFs ────────────────────────────────────────
    if zero_chunk_pdfs:
        print(f"\n  PDFs producing 0 real chunks ({len(zero_chunk_pdfs)} shown):")
        for r in zero_chunk_pdfs[:5]:
            print(f"    {r['name']}")
            print(f"      pages={r['pages']}  page_types={dict(r['page_types'])}")

    # ── Quick page-classifier test ─────────────────────────────────
    if zero_chunk_pdfs:
        print("\n  Running page classifier on first zero-chunk PDF...")
        zp = zero_chunk_pdfs[0]["path"]
        counts = classify_only(zp)
        print(f"  {Path(zp).name}: {counts}")

    if show_fix:
        _show_config_values()

    print("\n" + "═" * 65 + "\n")


def _show_config_values():
    print("\n  Current config values:")
    try:
        import config
        for attr in ["CHUNK_SIZE", "CHUNK_OVERLAP", "EMBED_MODEL",
                     "COLLECTION_NAME", "PDF_DIR"]:
            print(f"    {attr} = {getattr(config, attr, 'MISSING')}")
    except Exception as e:
        print(f"    Could not import config: {e}")


def quick_chunk_test():
    """Test chunking logic independently of OCR."""
    from config import CHUNK_SIZE, CHUNK_OVERLAP
    from ingestion.pdf_processor import chunk_text

    sample = "word " * 500
    chunks = chunk_text(sample)
    print(f"\n  Chunk test: 500 words → {len(chunks)} chunks")
    print(f"  CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")
    if len(chunks) == 0:
        print("  ❌ CRITICAL: chunk_text returned 0 chunks from 500 words!")
        print("     Check config.py CHUNK_SIZE and CHUNK_OVERLAP values.")
    elif len(chunks) == 1 and CHUNK_SIZE < 100:
        print("  ⚠ Only 1 chunk from 500 words — CHUNK_SIZE may be too large")
    else:
        print("  ✓ Chunking looks correct")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal SLM extraction diagnostics")
    parser.add_argument("--n",   type=int, default=20,  help="Number of random PDFs to test")
    parser.add_argument("--pdf", type=str, default=None, help="Test a specific PDF path")
    parser.add_argument("--fix", action="store_true",   help="Show current config values")
    parser.add_argument("--chunk-test", action="store_true", help="Test chunking logic only")
    args = parser.parse_args()

    if args.chunk_test:
        quick_chunk_test()
    else:
        run_diagnosis(n=args.n, specific_pdf=args.pdf, show_fix=args.fix)
