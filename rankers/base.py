import re
from abc import ABC, abstractmethod


class BaseRanker(ABC):
    name = ""
    label = ""

    def __init__(self, corpus):
        self.corpus = corpus

    @classmethod
    def is_available(cls):
        return True

    @abstractmethod
    def rank(self, query, top_k=10):
        raise NotImplementedError


def simple_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())
