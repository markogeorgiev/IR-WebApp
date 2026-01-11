import math
from collections import Counter

from rankers.base import BaseRanker, simple_tokenize


class QLDirichletRanker(BaseRanker):
    name = "ql-dirichlet"
    label = "Query Likelihood (Dirichlet)"

    def __init__(self, corpus, mu=1500):
        super().__init__(corpus)
        self._mu = mu
        self._doc_tokens = []
        self._doc_lengths = []
        collection = Counter()
        total_terms = 0
        for doc in corpus:
            tokens = simple_tokenize(doc["text"])
            self._doc_tokens.append(Counter(tokens))
            doc_len = len(tokens)
            self._doc_lengths.append(doc_len)
            collection.update(tokens)
            total_terms += doc_len
        self._collection = collection
        self._total_terms = max(total_terms, 1)

    def _collection_prob(self, term):
        return self._collection.get(term, 0) / self._total_terms

    def rank(self, query, top_k=10):
        terms = simple_tokenize(query)
        scores = []
        for idx, doc_tf in enumerate(self._doc_tokens):
            doc_len = self._doc_lengths[idx]
            score = 0.0
            for term in terms:
                p_wc = self._collection_prob(term)
                tf = doc_tf.get(term, 0)
                denom = doc_len + self._mu
                if denom == 0:
                    continue
                prob = (tf + self._mu * p_wc) / denom
                if prob > 0:
                    score += math.log(prob)
            scores.append(score)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            doc = self.corpus[idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": float(score),
                }
            )
        return results
