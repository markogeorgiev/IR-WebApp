import logging
import os
from pathlib import Path

import ir_datasets
import pandas as pd

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("prepare_corpus")


def normalize(text):
    if text is None:
        return ""
    return " ".join(text.lower().split())


def main():
    corpus_dir = Path(os.environ.get("CORPUS_DIR", "data"))
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_path = corpus_dir / "nfcorpus.csv"
    if out_path.exists():
        LOGGER.info("Corpus already exists at %s", out_path)
        return

    ds = ir_datasets.load("nfcorpus")

    rows = []
    for doc in ds.docs_iter():
        title = doc.title or ""
        abstract = doc.abstract or ""
        rows.append(
            {
                "doc_id": doc.doc_id,
                "title": title,
                "abstract": abstract,
                "content_key": normalize(title) + "||" + normalize(abstract),
                "url": doc.url,
            }
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="content_key", keep="first")
    df = df.drop(columns=["content_key"])
    df.to_csv(out_path, index=False)

    LOGGER.info("Number of entries after deduplication: %s", len(df))
    LOGGER.info("Corpus saved to %s", out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("prepare_corpus failed")
