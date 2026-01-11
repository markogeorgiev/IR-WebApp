import hashlib
import json
import os

import numpy as np

from rankers.base import BaseRanker

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class SBERTRanker(BaseRanker):
    name = "sbert"
    label = "SBERT"

    @classmethod
    def is_available(cls):
        return SentenceTransformer is not None

    def __init__(self, corpus, model_name=None):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        super().__init__(corpus)
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "data/cache")
        os.makedirs(cache_dir, exist_ok=True)
        fingerprint = self._corpus_fingerprint(corpus)
        model_slug = self._model_name.replace("/", "_")
        cache_base = os.path.join(cache_dir, f"sbert_{model_slug}_{fingerprint}")
        cache_meta = f"{cache_base}.json"
        cache_data = f"{cache_base}.npy"

        if os.path.exists(cache_meta) and os.path.exists(cache_data):
            with open(cache_meta, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("doc_ids") == [doc["id"] for doc in corpus]:
                self._embeddings = np.load(cache_data)
                return

        texts = [doc["text"] for doc in corpus]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)
        with open(cache_meta, "w", encoding="utf-8") as handle:
            json.dump({"doc_ids": [doc["id"] for doc in corpus]}, handle)
        np.save(cache_data, self._embeddings)

    def _corpus_fingerprint(self, corpus):
        digest = hashlib.sha256()
        for doc in corpus:
            digest.update(str(doc.get("id", "")).encode("utf-8"))
            digest.update(str(doc.get("title", "")).encode("utf-8"))
            digest.update(str(len(doc.get("text", ""))).encode("utf-8"))
        return digest.hexdigest()[:16]

    def rank(self, query, top_k=10):
        query_vec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self._embeddings, query_vec)
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in indices:
            doc = self.corpus[idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": float(scores[idx]),
                }
            )
        return results


class SBERTMiniLMRanker(SBERTRanker):
    name = "sbert-minilm-v6"
    label = "SBERT MiniLM-v6"

    def __init__(self, corpus, model_name=None):
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        super().__init__(corpus, model_name=model_name)
