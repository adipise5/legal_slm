"""
app.py — Flask backend for Legal SLM frontend
Run: python app.py
Open: http://localhost:5000
"""
from __future__ import annotations

import logging
import os
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, session
from flask_cors import CORS

app = Flask(__name__, static_folder="frontend", static_url_path="")
app.secret_key = os.urandom(32)
CORS(app, supports_credentials=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Auth ──────────────────────────────────────────────────────────────
CREDENTIALS = {"admin": "admin123"}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    uid  = data.get("username", "").strip()
    pwd  = data.get("password", "")
    if CREDENTIALS.get(uid) == pwd:
        session["logged_in"] = True
        session["user"]      = uid
        return jsonify({"ok": True, "user": uid})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me")
def me():
    if session.get("logged_in"):
        return jsonify({"logged_in": True, "user": session["user"]})
    return jsonify({"logged_in": False})


# ── Pipeline singleton ────────────────────────────────────────────────
_pipeline = None
_pipeline_error = None


def _get_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline, None
    if _pipeline_error:
        return None, _pipeline_error
    try:
        from pipeline import LegalSLMPipeline
        _pipeline = LegalSLMPipeline()
        return _pipeline, None
    except Exception as exc:
        _pipeline_error = str(exc)
        logger.error("[Pipeline] Init failed: %s", exc, exc_info=True)
        return None, _pipeline_error


# ── Query ─────────────────────────────────────────────────────────────
@app.route("/api/query", methods=["POST"])
@login_required
def query():
    data       = request.json or {}
    question   = data.get("question", "").strip()
    doc_filter = data.get("doc_type") or None

    if not question:
        return jsonify({"error": "Empty question"}), 400

    pipeline, err = _get_pipeline()

    if pipeline is None:
        # Pipeline not ready — return clear error (no fake demo data)
        return jsonify({
            "query":   question,
            "answer":  (
                f"⚠️ **System not ready**: {err}\n\n"
                "Possible causes:\n"
                "1. Run ingestion first: `python -m ingestion.parallel_ingestor`\n"
                "2. Start Ollama: `ollama serve` then `ollama pull mistral:7b-instruct-q4_K_M`\n"
                "3. Check that all retrieval/ and generation/ files are in place."
            ),
            "sources": [],
            "mode":    "error",
        })

    try:
        result = pipeline.query(question=question, doc_type_filter=doc_filter)
        return jsonify(result)
    except Exception as exc:
        logger.error("[Query] Failed: %s", exc, exc_info=True)
        return jsonify({
            "query":   question,
            "answer":  f"⚠️ Query error: {exc}",
            "sources": [],
            "mode":    "error",
        }), 500


# ── Stats ─────────────────────────────────────────────────────────────
@app.route("/api/stats")
@login_required
def stats():
    try:
        from qdrant_client import QdrantClient
        from config import QDRANT_PATH, COLLECTION_NAME, BM25_INDEX_PATH
        qdrant    = QdrantClient(path=QDRANT_PATH)
        vec_count = qdrant.count(COLLECTION_NAME).count if qdrant.collection_exists(COLLECTION_NAME) else 0
        bm25_ok   = Path(BM25_INDEX_PATH).exists()
        return jsonify({
            "vectors":    vec_count,
            "bm25_ready": bm25_ok,
            "status":     "ready" if vec_count > 0 else "empty",
        })
    except Exception as exc:
        return jsonify({"vectors": 0, "bm25_ready": False, "status": "empty", "error": str(exc)})


# ── Frontend ──────────────────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path=""):
    return send_from_directory("frontend", "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")