import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rankers.base import BaseRanker


class TFIDFRanker(BaseRanker):
    name = "tfidf"
    label = "TF-IDF"

    def __init__(self, corpus):
        super().__init__(corpus)
        texts = [doc["text"] for doc in corpus]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(texts)

    def rank(self, query, top_k=10):
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
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
