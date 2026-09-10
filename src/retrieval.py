"""Historical case retrieval using sentence embeddings."""
import numpy as np
import pandas as pd
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RetrievalIndex:
    """Simple embedding-based retrieval over historical cases."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str = "embeddings_cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._embeddings = None
        self._cases = None

    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, using TF-IDF fallback")
                self._model = "tfidf"

    def _get_cache_path(self, cases: list[dict]) -> Path:
        """Generate a deterministic cache path based on case IDs."""
        case_ids = sorted([c.get("case_id", "") for c in cases])
        content = "|".join(case_ids)
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        return self.cache_dir / f"embeddings_{hash_val}.npy"

    def build_index(self, cases: list[dict]):
        """Build the retrieval index from historical cases."""
        self._cases = cases
        cache_path = self._get_cache_path(cases)

        if cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            self._embeddings = np.load(cache_path)
            return

        self._load_model()
        texts = [self._format_case_text(c) for c in cases]

        if self._model == "tfidf":
            self._build_tfidf_index(texts)
        else:
            self._embeddings = self._model.encode(texts, show_progress_bar=True)

        # Cache embeddings
        np.save(cache_path, self._embeddings)
        logger.info(f"Built index with {len(cases)} cases, cached to {cache_path}")

    def _build_tfidf_index(self, texts: list[str]):
        """Fallback TF-IDF index."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self._embeddings = self._vectorizer.fit_transform(texts)

    def _format_case_text(self, case: dict) -> str:
        """Format a case for embedding."""
        parts = []
        if case.get("conversation_context"):
            parts.append(f"Context: {case['conversation_context']}")
        parts.append(f"Customer: {case['customer_text']}")
        return " ".join(parts)

    def retrieve(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Retrieve top K similar cases.

        Returns list of dicts with case_id, similarity, and case data.
        """
        if self._cases is None or self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index first.")

        self._load_model()

        if self._model == "tfidf":
            return self._retrieve_tfidf(query_text, top_k)

        # Encode query
        query_embedding = self._model.encode([query_text])

        # Compute similarities
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()

        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            case = self._cases[idx].copy()
            case["similarity"] = float(similarities[idx])
            results.append(case)

        return results

    def _retrieve_tfidf(self, query_text: str, top_k: int) -> list[dict]:
        """TF-IDF fallback retrieval."""
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self._embeddings).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            case = self._cases[idx].copy()
            case["similarity"] = float(similarities[idx])
            results.append(case)

        return results
