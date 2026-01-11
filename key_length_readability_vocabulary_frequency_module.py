import re

from textstat import textstat


def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def avg_sentence_length(text: str):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    total_words = sum(len(tokenize(s)) for s in sentences)
    return total_words / len(sentences)


def type_token_ratio(text: str):
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def analyze_text(text: str):
    tokens = tokenize(text)

    try:
        flesch = textstat.flesch_reading_ease(text)
    except Exception:
        flesch = None

    return {
        "token_count": len(tokens),
        "sentence_count": textstat.sentence_count(text),
        "avg_sentence_length": avg_sentence_length(text),
        "avg_word_length": (
            sum(len(w) for w in tokens) / len(tokens) if tokens else 0.0
        ),
        "type_token_ratio": type_token_ratio(text),
        "flesch_reading_ease": flesch,
    }
