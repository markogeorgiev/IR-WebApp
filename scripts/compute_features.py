import logging
import re
import time
from pathlib import Path

import pandas as pd
from textstat import textstat

from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("compute_features")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR")

CORPUS_PATH = PROJECT_ROOT / os.getenv("CORPUS_PATH")
OUTPUT_PATH = DATA_DIR / "nfcorpus_doc_klrvf.csv"


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


def main():
    df = pd.read_csv(CORPUS_PATH)

    rows = []
    processed = 0
    start = time.time()
    total = len(df)

    for _, row in df.iterrows():
        title = row["title"] if pd.notna(row["title"]) else ""
        abstract = row["abstract"] if pd.notna(row["abstract"]) else ""

        full_text = f"{title}\n{abstract}".strip()
        if not full_text:
            continue

        feats = analyze_text(full_text)

        rows.append(
            {
                "doc_id": row["doc_id"],
                "url": row["url"],
                "title": title,
                "abstract": abstract,
                **feats,
            }
        )

        processed += 1
        if processed % 1000 == 0:
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed else 0
            remaining = total - processed
            eta = remaining / rate if rate else 0
            LOGGER.info(
                "Processed %s/%s docs (%.2f docs/s, ETA %.1f min)",
                processed,
                total,
                rate,
                eta / 60,
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    LOGGER.info(
        "KLRVF analysis completed for %s documents. Saved to %s",
        processed,
        OUTPUT_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("compute_features failed")
