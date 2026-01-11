import numpy as np
from rank_bm25 import BM25Okapi

from rankers.base import BaseRanker, simple_tokenize


class BM25Ranker(BaseRanker):
    name = "bm25"
    label = "BM25"

    def __init__(self, corpus):
        super().__init__(corpus)
        self._tokenized = [simple_tokenize(doc["text"]) for doc in corpus]
        self._model = BM25Okapi(self._tokenized)

    def rank(self, query, top_k=10):
        scores = self._model.get_scores(simple_tokenize(query))
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
