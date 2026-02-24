"""
generation/llm_chain.py

Two generation modes:
  1. generate()            — grounded answer from indexed PDFs with citations
  2. generate_general_ai() — Ollama general knowledge fallback, clearly labelled,
                             no document references (web links acceptable)
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_MODEL   = "mistral:7b-instruct-q4_K_M"
OLLAMA_BASE    = "http://localhost:11434"

# Prompts ────────────────────────────────────────────────────────────

_GROUNDED_SYSTEM = """You are a precise legal assistant for Indian law enforcement.
You answer questions ONLY using the provided document excerpts.
Rules:
- Cite every fact as [SOURCE: filename, Page N]
- If a fact is not in the excerpts, say "Not found in provided documents."
- Be concise and factual. No speculation.
- Use plain English. Avoid legalese unless quoting directly."""

_GROUNDED_TEMPLATE = """Answer the question using ONLY the excerpts below.

QUESTION: {question}

DOCUMENT EXCERPTS:
{context}

Answer (cite sources inline as [SOURCE: filename, Page N]):"""


_GENERAL_SYSTEM = """You are a knowledgeable legal assistant specialising in Indian law.
You are answering from your general training knowledge because no matching documents were found in the local index.
Be helpful, accurate, and mention relevant IPC sections or legal provisions where applicable.
You may reference public legal resources."""

_GENERAL_TEMPLATE = """Answer the following legal question from your general knowledge.

QUESTION: {question}

Answer:"""


class LegalLLMChain:
    def __init__(self):
        self._client = None
        self._loaded = False

    def _get_client(self):
        if self._loaded:
            return self._client
        try:
            from ollama import Client
            self._client = Client(host=OLLAMA_BASE)
            # Quick connectivity test
            self._client.list()
            logger.info("[LLM] Ollama connected at %s", OLLAMA_BASE)
        except Exception as exc:
            logger.warning("[LLM] Ollama not reachable: %s", exc)
            self._client = None
        self._loaded = True
        return self._client

    # ── Grounded generation ──────────────────────────────────────────

    def generate(self, question: str, top_chunks: list[dict]) -> dict:
        """
        Generate an answer grounded in retrieved document chunks.
        Returns dict with: answer, sources, mode="document"
        """
        if not top_chunks:
            return self.generate_general_ai(question)

        # Build context block
        context_parts = []
        for i, c in enumerate(top_chunks, 1):
            context_parts.append(
                f"[{i}] File: {c['source_file']} | Page: {c['page_number']} "
                f"| Type: {c['doc_type']}\n{c['text']}"
            )
        context = "\n\n".join(context_parts)

        prompt = _GROUNDED_TEMPLATE.format(
            question=question,
            context=context,
        )

        answer = self._call_ollama(
            system=_GROUNDED_SYSTEM,
            prompt=prompt,
            max_tokens=1024,
        )

        # Build source list for the frontend
        sources = []
        for i, c in enumerate(top_chunks, 1):
            sources.append({
                "rank":         i,
                "source_file":  c["source_file"],
                "page_number":  c["page_number"],
                "doc_type":     c["doc_type"],
                "rerank_score": round(c.get("rerank_score", c.get("_score", 0.0)), 4),
                "preview":      c["text"][:200].replace("\n", " "),
            })

        return {
            "query":   question,
            "answer":  answer,
            "sources": sources,
            "mode":    "document",
        }

    # ── General AI fallback ──────────────────────────────────────────

    def generate_general_ai(self, question: str) -> dict:
        """
        Answer from Ollama general knowledge.
        Clearly labelled as AI-generated, no document sources attached.
        """
        prompt = _GENERAL_TEMPLATE.format(question=question)

        raw_answer = self._call_ollama(
            system=_GENERAL_SYSTEM,
            prompt=prompt,
            max_tokens=800,
        )

        # Prepend clear AI disclaimer
        answer = (
            "⚠️ **AI General Knowledge Answer** — "
            "No matching documents were found in your indexed PDFs. "
            "This answer is based on Mistral 7B's training knowledge and may not "
            "reflect the specific facts in your case files.\n\n"
            + raw_answer
        )

        return {
            "query":   question,
            "answer":  answer,
            "sources": [],          # No document sources for AI fallback
            "mode":    "ai_fallback",
        }

    # ── Ollama call ──────────────────────────────────────────────────

    def _call_ollama(
        self,
        system:     str,
        prompt:     str,
        max_tokens: int = 1024,
    ) -> str:
        client = self._get_client()

        if client is None:
            return (
                "[Ollama is not running. Start it with: ollama serve]\n\n"
                "To answer your question, please ensure Ollama is running "
                f"and the model '{OLLAMA_MODEL}' is available.\n"
                "Run: ollama pull mistral:7b-instruct-q4_K_M"
            )

        try:
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": prompt},
                ],
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.1,   # Low temp for factual legal answers
                    "top_p": 0.9,
                },
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            logger.error("[LLM] Ollama call failed: %s", exc)
            return (
                f"[LLM Error: {exc}]\n\n"
                "Could not generate answer. Check that Ollama is running:\n"
                "  brew services start ollama\n"
                "  ollama pull mistral:7b-instruct-q4_K_M"
            )