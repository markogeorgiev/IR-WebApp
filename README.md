# NF-Corpus Ranking WebApp

This is a small, modular webapp for experimenting with ranking models on NF-Corpus-style queries.

## Features
- Upload queries via text input, JSON, or CSV.
- Use bundled default queries.
- Choose between BM25, TF-IDF, and SBERT.
- Returns the top ranked documents for a query.

## Quick start (Docker)
```
docker compose up --build
```

Open `http://localhost:8000` (frontend).

UI pages:
- `/index.html` home
- `/models.html` model ranking
- `/queries.html` query dashboard
- `/dashboard.html` rankings + analysis
- `/doc_editor.html` document improvement

## Dataset inputs
- Default queries live in `data/default_queries.json`.
- The NF-Corpus docs are downloaded into a Docker volume at `/app/data/nfcorpus.csv`.
- The first run triggers `scripts/prepare_corpus.py` which downloads from `ir_datasets` and deduplicates.
- To use your own corpus, mount a JSON or CSV file and set `CORPUS_PATH`.

## Persistence
- Rankings are stored under `/app/data/rankings` (volume-backed).
- SBERT embeddings cache under `/app/data/cache` (volume-backed).
## Analysis plots
- Use the dashboard at `/dashboard.html` to generate rankings and plots on demand.
- Plots render dynamically based on what rankings and feature inputs exist.
- Feature plots require `data/nfcorpus_doc_klrvf.csv` mounted into the `data` volume.
  - Set `FRRA_N_JOBS=1` to avoid multiprocessing issues on slim images.
  - Set `FRRA_N_REPEATS` to reduce permutation importance time if needed.

## Containers
- `backend` runs the Flask API for ranking, rankings generation, and plots.
- `frontend` serves the static UI and proxies `/api` and `/plots` to the backend.

Expected corpus format (JSON):
```
[
  {"id": "D1", "title": "Title", "text": "Document text"},
  {"id": "D2", "title": "Title", "text": "Document text"}
]
```

Expected corpus format (CSV):
```
doc_id,title,abstract,url
MED-1,Title,Abstract text,https://example.com
```

## SBERT notes
The first run downloads the SBERT model. If you need to avoid downloads, pre-populate the Hugging Face cache or set `SBERT_MODEL` to a local model path.
