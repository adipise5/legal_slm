"""
Local embedding model wrapper.
Uses nomic-embed-text-v1.5 — strong performance on long legal text,
runs fully on Apple Silicon via MPS (Metal Performance Shaders).
"""
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL


class Embedder:
    def __init__(self):
        # Detect Apple Silicon MPS for hardware acceleration
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"[Embedder] Loading {EMBED_MODEL} on {self.device}")
        self.model = SentenceTransformer(
            EMBED_MODEL,
            device=self.device,
            trust_remote_code=True  # Required for nomic-embed-text
        )
    
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode texts to embeddings.
        normalize_embeddings=True is required for cosine similarity in Qdrant.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )