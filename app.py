import csv
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from rankers.registry import build_ranker, clear_cache, get_ranker, list_rankers

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
QUERIES_PATH = os.environ.get("QUERIES_PATH") or os.path.join(
    DATA_DIR, "queries.json"
)


@dataclass
class QueryStore:
    queries: list = field(default_factory=list)
    source: str = "default"
    updated_at: float = field(default_factory=time.time)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_queries(payload):
    normalized = []
    if isinstance(payload, list):
        for idx, item in enumerate(payload, start=1):
            if isinstance(item, str):
                normalized.append({"id": str(idx), "text": item})
            elif isinstance(item, dict):
                text = item.get("text") or item.get("query") or ""
                if text:
                    normalized.append(
                        {"id": str(item.get("id", idx)), "text": text}
                    )
    return normalized


def parse_text_queries(text_blob):
    queries = []
    for idx, line in enumerate(text_blob.splitlines(), start=1):
        line = line.strip()
        if line:
            queries.append({"id": str(idx), "text": line})
    return queries


def load_default_queries():
    path = os.path.join(DATA_DIR, "default_queries.json")
    payload = load_json(path)
    return normalize_queries(payload)


def load_saved_queries():
    if not os.path.exists(QUERIES_PATH):
        return None
    try:
        payload = load_json(QUERIES_PATH)
    except json.JSONDecodeError:
        return None
    queries = normalize_queries(payload)
    return queries if queries else None


def save_queries(queries):
    os.makedirs(os.path.dirname(QUERIES_PATH), exist_ok=True)
    with open(QUERIES_PATH, "w", encoding="utf-8") as handle:
        json.dump(queries, handle, ensure_ascii=False, indent=2)


def _load_corpus_json(corpus_path):
    payload = load_json(corpus_path)
    docs = []
    for idx, item in enumerate(payload, start=1):
        text = item.get("text") or ""
        title = item.get("title") or f"Doc {idx}"
        if text:
            docs.append(
                {
                    "id": str(item.get("id", idx)),
                    "title": title,
                    "text": text,
                }
            )
    return docs


def _load_corpus_csv(corpus_path):
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            doc_id = row.get("doc_id") or str(idx)
            title = row.get("title") or doc_id
            abstract = row.get("abstract") or ""
            text = f"Abstract: {abstract}\nText: {title}".strip()
            if text:
                docs.append({"id": doc_id, "title": title, "text": text})
    return docs


def load_corpus():
    corpus_path = os.environ.get("CORPUS_PATH") or os.path.join(
        DATA_DIR, "sample_corpus.json"
    )
    if os.path.exists(corpus_path) and corpus_path.lower().endswith(".csv"):
        return _load_corpus_csv(corpus_path)
    if os.path.exists(corpus_path):
        return _load_corpus_json(corpus_path)
    fallback_path = os.path.join(DATA_DIR, "sample_corpus.json")
    return _load_corpus_json(fallback_path)


def load_corpus_raw():
    corpus_path = os.environ.get("CORPUS_PATH")
    if not corpus_path:
        corpus_path = os.path.join(DATA_DIR, "sample_corpus.json")
    if os.path.exists(corpus_path) and corpus_path.lower().endswith(".csv"):
        raw = {}
        with open(corpus_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                doc_id = row.get("doc_id")
                if not doc_id:
                    continue
                raw[str(doc_id)] = {
                    "doc_id": str(doc_id),
                    "title": row.get("title") or "",
                    "abstract": row.get("abstract") or "",
                    "url": row.get("url") or "",
                }
        return raw
    return None


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = Flask(__name__)
LOGGER = logging.getLogger("webapp")
_saved_queries = load_saved_queries()
QUERY_STORE = QueryStore(
    queries=_saved_queries or load_default_queries(),
    source="saved" if _saved_queries else "default",
)
CORPUS = load_corpus()
CORPUS_RAW = load_corpus_raw()
DOC_INDEX = {doc["id"]: idx for idx, doc in enumerate(CORPUS)}
RANK_CACHE = {}
RANKINGS_CACHE = {}
ANALYSIS_STATE = {"status": "idle", "error": None, "last_run": None}
RANKING_STATE = {
    "status": "idle",
    "error": None,
    "last_run": None,
    "rows": 0,
    "last_model": None,
    "last_query": None,
}


def _startup_log():
    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "data/cache")
    cache_count = 0
    if os.path.isdir(cache_dir):
        cache_count = len([f for f in os.listdir(cache_dir) if f.endswith(".npy")])
    rankings_dir = _rankings_dir()
    rankings_count = len(
        [
            f
            for f in os.listdir(rankings_dir)
            if f.startswith("rankings_") and f.endswith(".csv")
        ]
    )
    plots_root = os.path.join(
        os.environ.get("PROJECT_ROOT", APP_DIR),
        os.environ.get("PLOTS_DIR", "outputs/plots"),
    )
    plots_count = 0
    if os.path.isdir(plots_root):
        for root, _, files in os.walk(plots_root):
            plots_count += len([f for f in files if f.endswith(".png")])
    feature_path = os.path.join(
        os.environ.get("PROJECT_ROOT", APP_DIR),
        os.environ.get("DATA_DIR", "data"),
        "nfcorpus_doc_klrvf.csv",
    )
    feature_status = "present" if os.path.exists(feature_path) else "missing"
    LOGGER.info(
        "Startup status: %s embedding cache files, %s rankings files, %s plot images, features %s.",
        cache_count,
        rankings_count,
        plots_count,
        feature_status,
    )


@app.route("/")
def index():
    return render_template(
        "index.html",
        query_count=len(QUERY_STORE.queries),
        source=QUERY_STORE.source,
        updated_at=QUERY_STORE.updated_at,
    )


@app.route("/models")
def models():
    return render_template("models.html")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/queries")
def queries_dashboard():
    return render_template("queries.html")


@app.route("/rankings")
def rankings_dashboard():
    return render_template("rankings.html")


@app.route("/plots/<path:filename>")
def plots(filename):
    plots_dir = os.environ.get("PLOTS_DIR", "outputs/plots")
    root = os.environ.get("PROJECT_ROOT", APP_DIR)
    return send_from_directory(os.path.join(root, plots_dir), filename)


def _run_analysis():
    ANALYSIS_STATE["status"] = "running"
    ANALYSIS_STATE["error"] = None
    try:
        env = os.environ.copy()
        env.setdefault("PROJECT_ROOT", APP_DIR)
        commands = [["python", "scripts/generate_rankings.py"]]
        if _rankings_available():
            commands.append(["python", "scripts/plot_agreement.py"])
        if _corpus_available() and not _feature_inputs_available():
            commands.append(["python", "scripts/compute_features.py"])
        if _feature_inputs_available():
            commands.append(["python", "scripts/plot_feature_analysis.py"])
        for command in commands:
            result = subprocess.run(command, env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(command)}")
        ANALYSIS_STATE["status"] = "completed"
        ANALYSIS_STATE["last_run"] = time.time()
    except Exception as exc:
        LOGGER.exception("Analysis job failed")
        ANALYSIS_STATE["status"] = "failed"
        ANALYSIS_STATE["error"] = str(exc)


@app.route("/api/analysis/run", methods=["POST"])
def run_analysis():
    if ANALYSIS_STATE["status"] == "running":
        return jsonify({"status": "running"})
    thread = threading.Thread(target=_run_analysis, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/analysis/status", methods=["GET"])
def analysis_status():
    return jsonify(ANALYSIS_STATE)




def _rankings_dir():
    root = os.environ.get("PROJECT_ROOT", APP_DIR)
    rel = os.environ.get("RANKINGS_DIR", "data/rankings")
    path = os.path.join(root, rel)
    os.makedirs(path, exist_ok=True)
    return path


_startup_log()


@app.errorhandler(Exception)
def handle_exception(error):
    LOGGER.exception("Unhandled exception on %s %s", request.method, request.path)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return "Internal server error", 500


def _rankings_available():
    rankings_dir = _rankings_dir()
    return any(
        name.startswith("rankings_") and name.endswith(".csv")
        for name in os.listdir(rankings_dir)
    )


def _feature_inputs_available():
    root = os.environ.get("PROJECT_ROOT", APP_DIR)
    data_dir = os.environ.get("DATA_DIR", "data")
    corpus_path = os.environ.get("CORPUS_PATH", os.path.join(data_dir, "nfcorpus.csv"))
    klrvf_path = os.path.join(root, data_dir, "nfcorpus_doc_klrvf.csv")
    corpus_full = corpus_path
    if not os.path.isabs(corpus_full):
        corpus_full = os.path.join(root, corpus_path)
    return os.path.exists(klrvf_path) and os.path.exists(corpus_full)


def _corpus_available():
    root = os.environ.get("PROJECT_ROOT", APP_DIR)
    data_dir = os.environ.get("DATA_DIR", "data")
    corpus_path = os.environ.get("CORPUS_PATH", os.path.join(data_dir, "nfcorpus.csv"))
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(root, corpus_path)
    return os.path.exists(corpus_path)


def _read_rankings_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _get_query_text(query_id):
    match = next((q for q in QUERY_STORE.queries if q["id"] == str(query_id)), None)
    return match["text"] if match else None


def _format_doc_text(title, abstract):
    return f"Abstract: {abstract}\nText: {title}".strip()


def _build_modified_corpus(doc_id, title, abstract):
    modified = []
    for doc in CORPUS:
        if doc["id"] == str(doc_id):
            modified.append(
                {
                    "id": doc["id"],
                    "title": title or doc.get("title", ""),
                    "text": _format_doc_text(title or "", abstract or ""),
                }
            )
        else:
            modified.append(doc)
    return modified


def _is_dense_model(model_name):
    return model_name in {"sbert-minilm-v6", "e5-small-v2"}


def _encode_passages(ranker, texts):
    if hasattr(ranker, "_format_passage"):
        texts = [ranker._format_passage(text) for text in texts]
    return ranker._model.encode(texts, normalize_embeddings=True)


def _encode_query(ranker, query_text):
    if hasattr(ranker, "_format_query"):
        query_text = ranker._format_query(query_text)
    return ranker._model.encode([query_text], normalize_embeddings=True)[0]


def _rank_dense_modified(model_name, query_text, edits):
    ranker = get_ranker(model_name, CORPUS)
    if not ranker:
        return None, None, "model not available"

    base_embeddings = ranker._embeddings
    updated = base_embeddings.copy()

    doc_ids = []
    texts = []
    for doc_id, edit in edits.items():
        idx = DOC_INDEX.get(doc_id)
        if idx is None:
            continue
        doc_ids.append(doc_id)
        texts.append(_format_doc_text(edit["title"], edit["abstract"]))

    if texts:
        new_vectors = _encode_passages(ranker, texts)
        for doc_id, vec in zip(doc_ids, new_vectors):
            idx = DOC_INDEX.get(doc_id)
            if idx is not None:
                updated[idx] = vec

    query_vec = _encode_query(ranker, query_text)
    scores = np.dot(updated, query_vec)
    order = np.argsort(scores)[::-1]

    new_rank_map = {}
    new_score_map = {}
    for rank, idx in enumerate(order, start=1):
        doc_id = CORPUS[idx]["id"]
        new_rank_map[doc_id] = rank
        new_score_map[doc_id] = float(scores[idx])
    return new_rank_map, new_score_map, None


def _modified_dir():
    root = os.environ.get("PROJECT_ROOT", APP_DIR)
    rel = os.environ.get("MODIFIED_DIR", "outputs/modified")
    path = os.path.join(root, rel)
    os.makedirs(path, exist_ok=True)
    return path


def _save_modified_doc(model_name, doc_id, title, abstract, url):
    out_path = os.path.join(_modified_dir(), f"edited_{model_name}.csv")
    rows = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    updated = False
    for row in rows:
        if row.get("doc_id") == str(doc_id):
            row["title"] = title
            row["abstract"] = abstract
            row["url"] = url
            updated = True
            break
    if not updated:
        rows.append(
            {
                "doc_id": str(doc_id),
                "title": title,
                "abstract": abstract,
                "url": url,
            }
        )
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["doc_id", "title", "abstract", "url"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_rankings_file(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
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
        for row in rows:
            writer.writerow(row)


def _rankings_file_path(model_name):
    return os.path.join(_rankings_dir(), f"rankings_{model_name}.csv")


def _load_rankings_cache(model_name):
    path = _rankings_file_path(model_name)
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    cached = RANKINGS_CACHE.get(model_name)
    if cached and cached.get("mtime") == mtime:
        return cached.get("by_query", {})
    rows = _read_rankings_file(path)
    by_query = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if not query_id:
            continue
        by_query.setdefault(query_id, []).append(row)
    RANKINGS_CACHE[model_name] = {"mtime": mtime, "by_query": by_query}
    return by_query


def _get_rankings_for_query(model_name, query_id):
    by_query = _load_rankings_cache(model_name)
    if not by_query:
        return []
    rows = by_query.get(str(query_id), [])
    return sorted(rows, key=lambda row: int(row.get("doc_rank", 0) or 0))


def _generate_rankings(model_name, queries):
    try:
        ranker = get_ranker(model_name, CORPUS)
    except Exception:
        LOGGER.exception("Ranker init failed for %s", model_name)
        return 0
    if not ranker:
        LOGGER.warning("Ranker not available: %s", model_name)
        return 0
    out_path = os.path.join(_rankings_dir(), f"rankings_{model_name}.csv")
    existing = _read_rankings_file(out_path)
    existing = [
        row for row in existing if row.get("query_id") not in {q["id"] for q in queries}
    ]
    rows = []
    top_k = len(CORPUS)
    for query in queries:
        try:
            results = ranker.rank(query["text"], top_k=top_k)
        except Exception:
            LOGGER.exception(
                "Ranking failed for model %s query %s", model_name, query.get("id")
            )
            continue
        for rank, item in enumerate(results, start=1):
            rows.append(
                {
                    "model": model_name,
                    "query_id": query["id"],
                    "doc_id": item["id"],
                    "doc_rank": rank,
                    "doc_score": item["score"],
                }
            )
    _write_rankings_file(out_path, existing + rows)
    return len(rows)


@app.route("/api/rankings/status", methods=["GET"])
def rankings_status():
    rankings_dir = _rankings_dir()
    files = [
        f
        for f in os.listdir(rankings_dir)
        if f.startswith("rankings_") and f.endswith(".csv")
    ]
    coverage = {}
    for filename in files:
        model_name = filename.replace("rankings_", "").replace(".csv", "")
        rows = _read_rankings_file(os.path.join(rankings_dir, filename))
        coverage[model_name] = sorted({row.get("query_id") for row in rows if row.get("query_id")})

    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "data/cache")
    cached_embeddings = []
    if os.path.isdir(cache_dir):
        for name in os.listdir(cache_dir):
            if (
                (name.startswith("sbert_") or name.startswith("e5_"))
                and name.endswith(".npy")
            ):
                cached_embeddings.append(name)

    return jsonify(
        {
            "models": list_rankers(),
            "queries": QUERY_STORE.queries,
            "coverage": coverage,
            "embedding_cache": {
                "count": len(cached_embeddings),
                "files": sorted(cached_embeddings),
            },
            "rankings_state": RANKING_STATE,
        }
    )


@app.route("/api/rankings/run", methods=["POST"])
def rankings_run():
    payload = request.get_json(silent=True) or {}
    model = payload.get("model")
    query_id = payload.get("query_id")

    if RANKING_STATE["status"] == "running":
        return jsonify({"status": "running"}), 202

    models = [m["name"] for m in list_rankers() if m["available"]]
    if model and model != "all":
        models = [model] if model in models else []
    if not models:
        return jsonify({"error": "model not available"}), 400

    queries = QUERY_STORE.queries
    if query_id and query_id != "all":
        match = next((q for q in queries if q["id"] == str(query_id)), None)
        if not match:
            return jsonify({"error": "query id not found"}), 404
        queries = [match]

    thread = threading.Thread(
        target=_run_rankings_job,
        args=(models, queries, model, query_id),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started"}), 202


def _run_rankings_job(models, queries, model_scope, query_scope):
    RANKING_STATE["status"] = "running"
    RANKING_STATE["error"] = None
    RANKING_STATE["rows"] = 0
    RANKING_STATE["last_model"] = model_scope
    RANKING_STATE["last_query"] = query_scope
    try:
        total_rows = 0
        for model_name in models:
            total_rows += _generate_rankings(model_name, queries)
        RANKING_STATE["rows"] = total_rows
        RANKING_STATE["status"] = "completed"
        RANKING_STATE["last_run"] = time.time()
    except Exception as exc:
        LOGGER.exception("Rankings run failed")
        RANKING_STATE["status"] = "failed"
        RANKING_STATE["error"] = str(exc)


@app.route("/api/analysis/plots", methods=["POST"])
def analysis_plots():
    if ANALYSIS_STATE["status"] == "running":
        return jsonify({"status": "running"})
    thread = threading.Thread(target=_run_plots_only, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


def _run_plots_only():
    ANALYSIS_STATE["status"] = "running"
    ANALYSIS_STATE["error"] = None
    try:
        env = os.environ.copy()
        env.setdefault("PROJECT_ROOT", APP_DIR)
        commands = []
        if _rankings_available():
            commands.append(["python", "scripts/plot_agreement.py"])
        if _corpus_available() and not _feature_inputs_available():
            commands.append(["python", "scripts/compute_features.py"])
        if _feature_inputs_available():
            commands.append(["python", "scripts/plot_feature_analysis.py"])
        for command in commands:
            result = subprocess.run(command, env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Command failed: {' '.join(command)}")
        ANALYSIS_STATE["status"] = "completed"
        ANALYSIS_STATE["last_run"] = time.time()
    except Exception as exc:
        LOGGER.exception("Plot generation failed")
        ANALYSIS_STATE["status"] = "failed"
        ANALYSIS_STATE["error"] = str(exc)


@app.route("/api/cache/embeddings/clear", methods=["POST"])
def clear_embedding_cache():
    root = os.environ.get("PROJECT_ROOT", APP_DIR)
    cache_dir = os.environ.get("EMBEDDING_CACHE_DIR", "data/cache")
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(root, cache_dir)
    removed = 0
    if os.path.isdir(cache_dir):
        for root, dirs, files in os.walk(cache_dir, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    continue
            for name in dirs:
                path = os.path.join(root, name)
                try:
                    os.rmdir(path)
                except OSError:
                    continue
    clear_cache()
    RANK_CACHE.clear()
    return jsonify({"status": "cleared", "removed_files": removed})


@app.route("/api/queries", methods=["GET", "POST"])
def queries():
    global QUERY_STORE
    if request.method == "GET":
        return jsonify(
            {
                "queries": QUERY_STORE.queries,
                "count": len(QUERY_STORE.queries),
                "source": QUERY_STORE.source,
                "updated_at": QUERY_STORE.updated_at,
            }
        )

    source = request.form.get("source") or "text"
    if source == "default":
        QUERY_STORE = QueryStore(queries=load_default_queries(), source="default")
        save_queries(QUERY_STORE.queries)
        return jsonify({"status": "ok", "count": len(QUERY_STORE.queries)})

    if "file" in request.files:
        file_obj = request.files["file"]
        content = file_obj.read().decode("utf-8", errors="ignore")
        if file_obj.filename.lower().endswith(".json"):
            payload = json.loads(content)
            queries = normalize_queries(payload)
        elif file_obj.filename.lower().endswith(".csv"):
            queries = []
            for idx, line in enumerate(content.splitlines(), start=1):
                if idx == 1 and "text" in line.lower():
                    continue
                text = line.split(",", 1)[-1].strip()
                if text:
                    queries.append({"id": str(idx), "text": text})
        else:
            queries = parse_text_queries(content)
    else:
        queries = parse_text_queries(request.form.get("queries_text", ""))

    QUERY_STORE = QueryStore(queries=queries, source=source)
    save_queries(QUERY_STORE.queries)
    return jsonify({"status": "ok", "count": len(QUERY_STORE.queries)})


@app.route("/api/queries/add", methods=["POST"])
def add_query():
    global QUERY_STORE
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    next_id = str(max([int(q["id"]) for q in QUERY_STORE.queries] or [0]) + 1)
    QUERY_STORE.queries.append({"id": next_id, "text": text})
    QUERY_STORE.source = "edited"
    QUERY_STORE.updated_at = time.time()
    save_queries(QUERY_STORE.queries)
    return jsonify({"status": "ok", "count": len(QUERY_STORE.queries)})


@app.route("/api/queries/<query_id>", methods=["PUT", "DELETE"])
def update_query(query_id):
    global QUERY_STORE
    match = next((q for q in QUERY_STORE.queries if q["id"] == str(query_id)), None)
    if not match:
        return jsonify({"error": "query id not found"}), 404

    if request.method == "DELETE":
        QUERY_STORE.queries = [
            q for q in QUERY_STORE.queries if q["id"] != str(query_id)
        ]
        QUERY_STORE.source = "edited"
        QUERY_STORE.updated_at = time.time()
        save_queries(QUERY_STORE.queries)
        return jsonify({"status": "ok", "count": len(QUERY_STORE.queries)})

    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    match["text"] = text
    QUERY_STORE.source = "edited"
    QUERY_STORE.updated_at = time.time()
    save_queries(QUERY_STORE.queries)
    return jsonify({"status": "ok", "count": len(QUERY_STORE.queries)})


@app.route("/api/queries/reset", methods=["POST"])
def reset_queries():
    global QUERY_STORE
    QUERY_STORE = QueryStore(queries=load_default_queries(), source="default")
    save_queries(QUERY_STORE.queries)
    return jsonify({"status": "ok", "count": len(QUERY_STORE.queries)})


@app.route("/api/models", methods=["GET"])
def models_api():
    return jsonify({"models": list_rankers()})


@app.route("/api/docs/search", methods=["GET"])
def docs_search():
    query = (request.args.get("q") or "").strip().lower()
    if not query:
        return jsonify({"results": []})
    results = []
    if CORPUS_RAW:
        for doc_id, row in CORPUS_RAW.items():
            text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
            if query in doc_id.lower() or query in text:
                snippet = (row.get("abstract") or row.get("title") or "")[:180]
                results.append(
                    {
                        "doc_id": doc_id,
                        "title": row.get("title") or "",
                        "snippet": snippet,
                    }
                )
            if len(results) >= 25:
                break
    else:
        for doc in CORPUS:
            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            if query in doc.get("id", "").lower() or query in text:
                results.append(
                    {
                        "doc_id": doc.get("id"),
                        "title": doc.get("title", ""),
                        "snippet": doc.get("text", "")[:180],
                    }
                )
            if len(results) >= 25:
                break
    return jsonify({"results": results})


@app.route("/api/docs/<doc_id>", methods=["GET"])
def doc_details(doc_id):
    doc_id = str(doc_id)
    if CORPUS_RAW and doc_id in CORPUS_RAW:
        return jsonify(CORPUS_RAW[doc_id])
    idx = DOC_INDEX.get(doc_id)
    if idx is None:
        return jsonify({"error": "doc not found"}), 404
    doc = CORPUS[idx]
    return jsonify(
        {
            "doc_id": doc["id"],
            "title": doc.get("title") or "",
            "abstract": "",
            "url": "",
        }
    )


@app.route("/api/docs/rerank", methods=["POST"])
def doc_rerank():
    payload = request.get_json(silent=True) or {}
    doc_id = str(payload.get("doc_id") or "")
    model_name = payload.get("model")
    query_id = payload.get("query_id")
    title = payload.get("title") or ""
    abstract = payload.get("abstract") or ""

    if not doc_id or not model_name or not query_id:
        return jsonify({"error": "doc_id, model, and query_id are required"}), 400

    query_text = _get_query_text(query_id)
    if not query_text:
        return jsonify({"error": "query not found"}), 404

    if doc_id not in DOC_INDEX:
        return jsonify({"error": "doc not found"}), 404

    top_k = len(CORPUS)
    try:
        original_ranker = get_ranker(model_name, CORPUS)
        if not original_ranker:
            return jsonify({"error": "model not available"}), 400
        original_results = original_ranker.rank(query_text, top_k=top_k)
        old_rank = next(
            (idx + 1 for idx, item in enumerate(original_results) if item["id"] == doc_id),
            None,
        )
        old_score = next(
            (item["score"] for item in original_results if item["id"] == doc_id),
            None,
        )
    except Exception as exc:
        LOGGER.exception(
            "Original ranking failed for doc %s model %s", doc_id, model_name
        )
        return jsonify({"error": str(exc)}), 500

    new_rank = None
    new_score = None
    if _is_dense_model(model_name):
        edits = {doc_id: {"title": title, "abstract": abstract}}
        try:
            new_rank_map, new_score_map, error = _rank_dense_modified(
                model_name, query_text, edits
            )
        except Exception as exc:
            LOGGER.exception(
                "Modified ranking failed for doc %s model %s", doc_id, model_name
            )
            return jsonify({"error": str(exc)}), 500
        if error:
            return jsonify({"error": error}), 400
        new_rank = new_rank_map.get(doc_id)
        new_score = new_score_map.get(doc_id)
    else:
        modified_corpus = _build_modified_corpus(doc_id, title, abstract)
        try:
            modified_ranker = build_ranker(model_name, modified_corpus, use_cache=False)
            if not modified_ranker:
                return jsonify({"error": "model not available"}), 400
            modified_results = modified_ranker.rank(query_text, top_k=top_k)
            new_rank = next(
                (idx + 1 for idx, item in enumerate(modified_results) if item["id"] == doc_id),
                None,
            )
            new_score = next(
                (item["score"] for item in modified_results if item["id"] == doc_id),
                None,
            )
        except Exception as exc:
            LOGGER.exception(
                "Modified ranking failed for doc %s model %s", doc_id, model_name
            )
            return jsonify({"error": str(exc)}), 500

    url = ""
    if CORPUS_RAW and doc_id in CORPUS_RAW:
        url = CORPUS_RAW[doc_id].get("url") or ""
    _save_modified_doc(model_name, doc_id, title, abstract, url)

    plot_error = None
    try:
        env = os.environ.copy()
        env.setdefault("PROJECT_ROOT", APP_DIR)
        result = subprocess.run(
            ["python", "scripts/plot_klrvf_comparison.py"],
            env=env,
            cwd=APP_DIR,
            check=False,
        )
        if result.returncode != 0:
            plot_error = "plot generation failed"
    except Exception as exc:
        LOGGER.exception("KLRVF plot generation failed for doc %s", doc_id)
        plot_error = str(exc)

    plot_url = f"/plots/klrvf_comparisons/{doc_id}_{model_name}_comprehensive.png"
    response = {
        "old_rank": old_rank,
        "new_rank": new_rank,
        "rank_change": None if old_rank is None or new_rank is None else new_rank - old_rank,
        "old_score": old_score,
        "new_score": new_score,
        "plot_url": plot_url,
    }
    if plot_error:
        response["plot_error"] = plot_error
    return jsonify(response)


@app.route("/api/docs/rerank/batch", methods=["POST"])
def doc_rerank_batch():
    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model")
    query_id = payload.get("query_id")
    docs = payload.get("docs") or []

    if not model_name or not query_id:
        return jsonify({"error": "model and query_id are required"}), 400

    query_text = _get_query_text(query_id)
    if not query_text:
        return jsonify({"error": "query not found"}), 404

    if not isinstance(docs, list) or not docs:
        return jsonify({"error": "docs is required"}), 400
    if len({str(d.get("doc_id") or "") for d in docs}) != len(docs):
        return jsonify({"error": "duplicate doc_id in request"}), 400

    top_k = len(CORPUS)
    try:
        original_ranker = get_ranker(model_name, CORPUS)
    except Exception as exc:
        LOGGER.exception("Ranker init failed for model %s", model_name)
        return jsonify({"error": str(exc)}), 500
    if not original_ranker:
        return jsonify({"error": "model not available"}), 400

    try:
        original_results = original_ranker.rank(query_text, top_k=top_k)
    except Exception as exc:
        LOGGER.exception("Original ranking failed for model %s", model_name)
        return jsonify({"error": str(exc)}), 500

    old_rank_map = {}
    old_score_map = {}
    for idx, item in enumerate(original_results, start=1):
        old_rank_map[item["id"]] = idx
        old_score_map[item["id"]] = item["score"]

    results = []
    edits = {}
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        title = doc.get("title") or ""
        abstract = doc.get("abstract") or ""
        if not doc_id:
            results.append({"doc_id": doc_id, "error": "doc_id is required"})
            continue
        if doc_id not in DOC_INDEX:
            results.append({"doc_id": doc_id, "error": "doc not found"})
            continue
        edits[doc_id] = {"title": title, "abstract": abstract}

    if not edits:
        return jsonify({"error": "no valid docs to rerank"}), 400

    modified_corpus = []
    for doc in CORPUS:
        doc_id = str(doc["id"])
        if doc_id in edits:
            edit = edits[doc_id]
            modified_corpus.append(
                {
                    "id": doc["id"],
                    "title": edit["title"] or doc.get("title", ""),
                    "text": _format_doc_text(edit["title"] or "", edit["abstract"] or ""),
                }
            )
        else:
            modified_corpus.append(doc)

    if _is_dense_model(model_name):
        try:
            new_rank_map, new_score_map, error = _rank_dense_modified(
                model_name, query_text, edits
            )
        except Exception as exc:
            LOGGER.exception("Modified ranking failed for model %s", model_name)
            return jsonify({"error": str(exc)}), 500
        if error:
            return jsonify({"error": error}), 400
    else:
        try:
            modified_ranker = build_ranker(model_name, modified_corpus, use_cache=False)
        except Exception as exc:
            LOGGER.exception("Modified ranker init failed for model %s", model_name)
            return jsonify({"error": str(exc)}), 500
        if not modified_ranker:
            return jsonify({"error": "model not available"}), 400

        try:
            modified_results = modified_ranker.rank(query_text, top_k=top_k)
        except Exception as exc:
            LOGGER.exception("Modified ranking failed for model %s", model_name)
            return jsonify({"error": str(exc)}), 500

        new_rank_map = {}
        new_score_map = {}
        for idx, item in enumerate(modified_results, start=1):
            new_rank_map[item["id"]] = idx
            new_score_map[item["id"]] = item["score"]

    for doc_id, edit in edits.items():
        old_rank = old_rank_map.get(doc_id)
        old_score = old_score_map.get(doc_id)
        new_rank = new_rank_map.get(doc_id)
        new_score = new_score_map.get(doc_id)

        url = ""
        if CORPUS_RAW and doc_id in CORPUS_RAW:
            url = CORPUS_RAW[doc_id].get("url") or ""
        _save_modified_doc(model_name, doc_id, edit["title"], edit["abstract"], url)

        results.append(
            {
                "doc_id": doc_id,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "rank_change": None
                if old_rank is None or new_rank is None
                else new_rank - old_rank,
                "old_score": old_score,
                "new_score": new_score,
                "plot_url": f"/plots/klrvf_comparisons/{doc_id}_{model_name}_comprehensive.png",
            }
        )

    plot_error = None
    try:
        env = os.environ.copy()
        env.setdefault("PROJECT_ROOT", APP_DIR)
        result = subprocess.run(
            ["python", "scripts/plot_klrvf_comparison.py"],
            env=env,
            cwd=APP_DIR,
            check=False,
        )
        if result.returncode != 0:
            plot_error = "plot generation failed"
    except Exception as exc:
        LOGGER.exception("KLRVF plot generation failed for batch model %s", model_name)
        plot_error = str(exc)

    response = {"status": "ok", "results": results}
    if plot_error:
        response["plot_error"] = plot_error
    return jsonify(response)


@app.route("/api/rank", methods=["POST"])
def rank():
    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model")
    query_id = payload.get("query_id")
    query_text = payload.get("query_text")
    top_k_raw = payload.get("top_k", 10)
    if isinstance(top_k_raw, str) and top_k_raw.lower() == "all":
        top_k = len(CORPUS)
    else:
        top_k = int(top_k_raw)

    if not model_name:
        return jsonify({"error": "model is required"}), 400

    if query_id:
        match = next(
            (q for q in QUERY_STORE.queries if q["id"] == str(query_id)), None
        )
        if not match:
            return jsonify({"error": "query id not found"}), 404
        query_text = match["text"]

    if not query_text:
        return jsonify({"error": "query text is required"}), 400

    if query_id:
        cached_rows = _get_rankings_for_query(model_name, query_id)
        if cached_rows and len(cached_rows) >= len(CORPUS):
            results = []
            limit = len(CORPUS) if top_k == len(CORPUS) else top_k
            for row in cached_rows[:limit]:
                doc_id = str(row.get("doc_id") or "")
                idx = DOC_INDEX.get(doc_id)
                if idx is None:
                    continue
                doc = CORPUS[idx]
                try:
                    score = float(row.get("doc_score"))
                except (TypeError, ValueError):
                    score = None
                results.append(
                    {
                        "id": doc["id"],
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                        "score": score if score is not None else 0.0,
                    }
                )
            return jsonify({"query": query_text, "results": results})

    cache_key = (model_name, query_text, top_k)
    if cache_key in RANK_CACHE:
        return jsonify({"query": query_text, "results": RANK_CACHE[cache_key]})

    try:
        ranker = get_ranker(model_name, CORPUS)
    except Exception as exc:
        LOGGER.exception("Ranker init failed for model %s", model_name)
        return jsonify({"error": str(exc)}), 500
    if not ranker:
        return jsonify({"error": "model not available"}), 400

    try:
        results = ranker.rank(query_text, top_k=top_k)
    except Exception as exc:
        LOGGER.exception("Ranking failed for model %s", model_name)
        return jsonify({"error": str(exc)}), 500
    RANK_CACHE[cache_key] = results
    return jsonify({"query": query_text, "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
