import csv
import logging
import os
from pathlib import Path

from app import load_corpus, load_default_queries, load_saved_queries
from rankers.registry import get_ranker, list_rankers


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("generate_rankings")


def main():
    project_root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    rankings_dir = project_root / os.environ.get("RANKINGS_DIR", "data/rankings")
    rankings_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus()
    queries = load_saved_queries() or load_default_queries()

    corpus_size = len(corpus)
    for model in list_rankers():
        if not model["available"]:
            continue
        model_name = model["name"]
        out_path = rankings_dir / f"rankings_{model_name}.csv"
        if out_path.exists():
            LOGGER.info("Rankings already exist for %s", model_name)
            continue
        ranker = get_ranker(model_name, corpus)
        if not ranker:
            LOGGER.warning("Ranker not available: %s", model_name)
            continue

        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model",
                    "query_id",
                    "doc_id",
                    "doc_rank",
                    "doc_score",
                ],
            )
            writer.writeheader()
            for query in queries:
                results = ranker.rank(query["text"], top_k=corpus_size)
                for rank, item in enumerate(results, start=1):
                    writer.writerow(
                        {
                            "model": model_name,
                            "query_id": query["id"],
                            "doc_id": item["id"],
                            "doc_rank": rank,
                            "doc_score": item["score"],
                        }
                    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("generate_rankings failed")
