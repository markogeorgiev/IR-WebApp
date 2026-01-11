import os

from rankers.bm25 import BM25Ranker
from rankers.e5 import E5SmallRanker
from rankers.ql_dirichlet import QLDirichletRanker
from rankers.sbert import SBERTRanker, SBERTMiniLMRanker
from rankers.tfidf import TFIDFRanker

RANKER_CLASSES = [
    BM25Ranker,
    TFIDFRanker,
    SBERTRanker,
    SBERTMiniLMRanker,
    E5SmallRanker,
    QLDirichletRanker,
]
_CACHE = {}


def list_rankers():
    models = []
    for cls in RANKER_CLASSES:
        models.append(
            {
                "name": cls.name,
                "label": cls.label,
                "available": cls.is_available(),
            }
        )
    return models


def _cache_key(name):
    return name.lower()


def build_ranker(name, corpus, use_cache=True):
    key = _cache_key(name)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    for cls in RANKER_CLASSES:
        if cls.name == key:
            if not cls.is_available():
                return None
            if cls.name in {"sbert", "sbert-minilm-v6"}:
                model_name = os.environ.get("SBERT_MODEL")
                instance = cls(corpus, model_name=model_name)
            else:
                instance = cls(corpus)
            if use_cache:
                _CACHE[key] = instance
            return instance
    return None


def get_ranker(name, corpus):
    return build_ranker(name, corpus, use_cache=True)


def clear_cache():
    _CACHE.clear()
