# Legal SLM — Installation Guide

## Prerequisites
- **Python 3.10, 3.11, or 3.12** — do NOT use 3.13 (PaddlePaddle is incompatible)
- **Ollama** (for Mistral 7B generation)
- **Tesseract OCR** (for scanned document fallback)

---

## Step 1 — Install Tesseract OCR

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run `tesseract-ocr-w64-setup-5.x.x.exe`
3. During install, tick "Additional language data (download)" → select Hindi
4. Add Tesseract to PATH: `C:\Program Files\Tesseract-OCR`
5. Set environment variable: `TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata`

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng
```

---

## Step 2 — Install Ollama + Mistral

**macOS / Linux:**
```bash
brew install ollama          # macOS
# OR: curl -fsSL https://ollama.com/install.sh | sh   (Linux)

brew services start ollama   # macOS — auto-starts on login
# OR: ollama serve &          (Linux/manual)

ollama pull mistral:7b-instruct-q4_K_M
```

**Windows:**
1. Download from https://ollama.com/download/windows
2. Run installer — Ollama starts automatically as a background service
3. Open Command Prompt:
```cmd
ollama pull mistral:7b-instruct-q4_K_M
```

---

## Step 3 — Create Python virtual environment

**macOS / Linux:**
```bash
cd legal_slm
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
cd legal_slm
python -m venv .venv
.venv\Scripts\Activate.ps1
# If you get an execution policy error:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 4 — Install PyTorch (platform-specific)

### macOS Apple Silicon (M1/M2/M3/M4):
```bash
pip install torch==2.4.0
# MPS (Metal) acceleration is included — no extra steps needed
```

### macOS Intel:
```bash
pip install torch==2.4.0
# CPU-only, no GPU acceleration
```

### Windows (CPU-only — recommended for most setups):
```powershell
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### Windows (NVIDIA GPU):
```powershell
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### Linux (CPU):
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 5 — Install all Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `paddleocr` will automatically install `opencv-contrib-python==4.10.0.84`.
> Do NOT separately install `opencv-python` — it conflicts with paddleocr's opencv variant.

---

## Step 6 — Optional: M4 Air performance tuning

Apply these once to speed up Mistral on Apple Silicon:
```bash
sudo launchctl setenv OLLAMA_FLASH_ATTENTION 1
sudo launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
brew services restart ollama
```

---

## Step 7 — Place your PDFs

Drop all 26,000 PDFs into:
```
legal_slm/data/legal_pdfs/
```
Subfolders are supported — the ingestion scans recursively.

---

## Step 8 — Run ingestion

```bash
# First-time (or after adding new files):
python -m ingestion.parallel_ingestor

# If you want to wipe and re-ingest everything from scratch:
python -m ingestion.parallel_ingestor --reset

# To use fewer workers (e.g. use machine normally while ingesting):
python -m ingestion.parallel_ingestor --workers 2
```

Expected time for 2,000 PDFs: ~15–30 minutes depending on scan quality.
Expected time for 26,000 PDFs: ~2–4 hours with 4 workers.

---

## Step 9 — Start the web app

```bash
# Make sure Ollama is running first:
ollama serve    # skip this on macOS if using brew services

# Then:
python app.py
```

Open http://localhost:5000 — Login: `admin` / `admin123`

---

## Folder structure after setup

```
legal_slm/
├── data/
│   └── legal_pdfs/          ← all your PDFs here
├── state/                   ← auto-created during ingestion
│   ├── qdrant_db/
│   ├── bm25_index.pkl
│   ├── processed_hashes.json
│   └── ingestion_audit.jsonl
├── ingestion/
├── retrieval/
├── generation/
├── frontend/
├── app.py
├── pipeline.py
├── config.py
├── requirements.txt
└── INSTALL.md
```

---

## Troubleshooting

**"PaddleOCR not installed" warning:**
This is harmless if `PADDLE_AVAILABLE=True` appears in the logs.
The warning comes from an internal PaddleOCR connectivity check.

**"Cannot connect to Ollama":**
Run `ollama serve` in a separate terminal, or on macOS: `brew services start ollama`

**Windows: "OSError: [WinError 1455] The paging file is too small":**
This means RAM is exhausted. Reduce workers: `--workers 1`

**Windows: multiprocessing spawn errors:**
Add this to the top of any script you run directly:
```python
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
```

**"Index empty — run ingestion" on the web UI:**
The ingestion didn't complete or Qdrant has 0 vectors.
Check: `python -c "from qdrant_client import QdrantClient; c = QdrantClient(path='state/qdrant_db'); print(c.count('legal_docs'))"`
If 0, run ingestion again (it's incremental — already-done files are skipped).