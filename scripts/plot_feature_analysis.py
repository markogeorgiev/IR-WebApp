import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("plot_feature_analysis")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()

RANKINGS_DIR = PROJECT_ROOT / os.getenv("RANKINGS_DIR")
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR")

CORPUS_PATH = PROJECT_ROOT / os.getenv("CORPUS_PATH")
KLRVF_CSV = DATA_DIR / "nfcorpus_doc_klrvf.csv"

OUTPUTS_DIR = PROJECT_ROOT / os.getenv("OUTPUTS_DIR")
METRICS_DIR = OUTPUTS_DIR / os.getenv("METRICS_DIR")
PLOTS_DIR = PROJECT_ROOT  / os.getenv("PLOTS_DIR")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANKING_FILES = sorted(RANKINGS_DIR.glob("rankings_*.csv"))

OUT_CORR = METRICS_DIR / "frra_correlations.csv"
OUT_IMPORTANCE = METRICS_DIR / "frra_importance.csv"

PLOT_CORR = PLOTS_DIR / "feature_correlations.png"
PLOT_IMP = PLOTS_DIR / "feature_importances.png"


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return math.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return math.nan
    return float(spearmanr(x, y, nan_policy="omit").correlation)


def load_rankings(files):
    if not files:
        raise RuntimeError("No rankings_*.csv files found in rankings/")
    dfs = [pd.read_csv(f, dtype={"query_id": str}) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["doc_rank"] = pd.to_numeric(df["doc_rank"], errors="coerce")
    df["doc_score"] = pd.to_numeric(df["doc_score"], errors="coerce")
    return df


def load_features(klrvf_csv: Path, corpus_csv: Path):
    feat = pd.read_csv(klrvf_csv)
    corp = pd.read_csv(corpus_csv, usecols=["doc_id", "title", "abstract"])

    def _text_len(row):
        t = row["title"] if pd.notna(row["title"]) else ""
        a = row["abstract"] if pd.notna(row["abstract"]) else ""
        return len((t + "\n" + a).strip())

    corp["doc_length"] = corp.apply(_text_len, axis=1)

    feat = feat.merge(corp[["doc_id", "doc_length"]], on="doc_id", how="left")

    feat = feat.rename(
        columns={
            "token_count": "word_count",
            "avg_word_length": "avg_word_length",
            "flesch_reading_ease": "readability",
            "type_token_ratio": "lexical_diversity",
            "sentence_count": "sentence_count",
        }
    )

    feat["doc_length"] = pd.to_numeric(feat["doc_length"], errors="coerce")
    feat["word_count"] = pd.to_numeric(feat["word_count"], errors="coerce")
    feat["avg_word_length"] = pd.to_numeric(feat["avg_word_length"], errors="coerce")
    feat["readability"] = pd.to_numeric(feat["readability"], errors="coerce")
    feat["lexical_diversity"] = pd.to_numeric(feat["lexical_diversity"], errors="coerce")
    feat["sentence_count"] = pd.to_numeric(feat["sentence_count"], errors="coerce")

    feat = feat[["doc_id", "avg_word_length", "doc_length", "lexical_diversity", "readability", "sentence_count", "word_count"]]
    return feat


def analyze_correlations(merged_df, feature_columns):
    rows = []
    start = time.time()
    for model in sorted(merged_df["model"].unique()):
        mdf = merged_df[merged_df["model"] == model]
        for feat in feature_columns:
            valid = mdf[[feat, "doc_rank", "doc_score"]].dropna()
            corr_rank = safe_spearman(valid[feat], valid["doc_rank"]) if len(valid) >= 2 else math.nan
            corr_score = safe_spearman(valid[feat], valid["doc_score"]) if len(valid) >= 2 else math.nan
            rows.append(
                {
                    "model": model,
                    "feature": feat,
                    "corr_with_rank": corr_rank,
                    "corr_with_score": corr_score,
                    "n": int(len(valid)),
                }
            )
        elapsed = time.time() - start
        LOGGER.info(
            "Correlation pass for %s completed (%.1f min elapsed)",
            model,
            elapsed / 60,
        )
    return pd.DataFrame(rows)


def compute_importance(
    merged_df,
    feature_columns,
    target="doc_score",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
):
    rows = []
    start = time.time()
    for model in sorted(merged_df["model"].unique()):
        mdf = merged_df[merged_df["model"] == model].dropna(subset=feature_columns + [target])

        if len(mdf) < 50:
            continue

        X = mdf[feature_columns].astype(float)
        y = mdf[target].astype(float)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        rf = RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=n_jobs
        )
        rf.fit(Xs, y)

        perm = permutation_importance(
            rf, Xs, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
        )

        for i, feat in enumerate(feature_columns):
            rows.append(
                {
                    "model": model,
                    "feature": feat,
                    "rf_importance": float(rf.feature_importances_[i]),
                    "perm_importance_mean": float(perm.importances_mean[i]),
                    "perm_importance_std": float(perm.importances_std[i]),
                    "n": int(len(mdf)),
                    "target": target,
                }
            )
        elapsed = time.time() - start
        remaining = len(sorted(merged_df["model"].unique())) - (
            sorted(merged_df["model"].unique()).index(model) + 1
        )
        avg = elapsed / (sorted(merged_df["model"].unique()).index(model) + 1)
        eta = avg * remaining
        LOGGER.info(
            "Importance for %s done (%.1f min elapsed, ETA %.1f min)",
            model,
            elapsed / 60,
            eta / 60,
        )
    return pd.DataFrame(rows)


def plot_correlation_heatmaps(corr_df, out_path):
    pivot_rank = corr_df.pivot(index="feature", columns="model", values="corr_with_rank")
    pivot_score = corr_df.pivot(index="feature", columns="model", values="corr_with_score")

    # Clean feature names for display
    feature_labels = {
        "avg_word_length": "Avg Word Length",
        "doc_length": "Document Length",
        "lexical_diversity": "Lexical Diversity",
        "readability": "Readability",
        "sentence_count": "Sentence Count",
        "word_count": "Word Count"
    }
    pivot_rank.index = [feature_labels.get(f, f) for f in pivot_rank.index]
    pivot_score.index = [feature_labels.get(f, f) for f in pivot_score.index]

    sns.set_style("white")
    plt.rcParams["font.size"] = 10
    plt.rcParams['font.family'] = 'sans-serif'

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=150)

    # Rank correlation heatmap
    sns.heatmap(
        pivot_rank,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0.0,
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Spearman ρ"},
        ax=axes[0],
        square=False,
        annot_kws={"size": 9, "weight": "bold"}
    )
    axes[0].set_title(
        "Feature-Rank Correlations by Model\n(negative = feature associated with higher rank)",
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    axes[0].set_xlabel("Model", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Feature", fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=10)

    # Score correlation heatmap
    sns.heatmap(
        pivot_score,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0.0,
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Spearman ρ"},
        ax=axes[1],
        square=False,
        annot_kws={"size": 9, "weight": "bold"}
    )
    axes[1].set_title(
        "Feature-Score Correlations by Model\n(positive = feature associated with higher score)",
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    axes[1].set_xlabel("Model", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Feature", fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor='white', dpi=150)
    plt.show()
    plt.close(fig)


def plot_importance_heatmaps(imp_df, out_path):
    pivot_rf = imp_df.pivot(index="feature", columns="model", values="rf_importance")
    pivot_perm = imp_df.pivot(index="feature", columns="model", values="perm_importance_mean")

    # Clean feature names for display
    feature_labels = {
        "avg_word_length": "Avg Word Length",
        "doc_length": "Document Length",
        "lexical_diversity": "Lexical Diversity",
        "readability": "Readability",
        "sentence_count": "Sentence Count",
        "word_count": "Word Count"
    }
    pivot_rf.index = [feature_labels.get(f, f) for f in pivot_rf.index]
    pivot_perm.index = [feature_labels.get(f, f) for f in pivot_perm.index]

    sns.set_style("white")
    plt.rcParams["font.size"] = 10
    plt.rcParams['font.family'] = 'sans-serif'

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=150)

    # RF importance heatmap
    vmax_rf = pivot_rf.max().max()
    sns.heatmap(
        pivot_rf,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Importance", "shrink": 0.8},
        ax=axes[0],
        vmin=0,
        vmax=vmax_rf,
        square=False,
        annot_kws={"size": 9, "weight": "bold"}
    )
    axes[0].set_title(
        "Random Forest Feature Importance by Model\n(higher values = more predictive of document score)",
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    axes[0].set_xlabel("Model", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Feature", fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=10)

    # Permutation importance heatmap
    vmax_perm = pivot_perm.max().max()
    sns.heatmap(
        pivot_perm,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Importance", "shrink": 0.8},
        ax=axes[1],
        vmin=0,
        vmax=vmax_perm,
        square=False,
        annot_kws={"size": 9, "weight": "bold"}
    )
    axes[1].set_title(
        "Permutation Feature Importance by Model\n(decrease in model performance when feature is shuffled)",
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    axes[1].set_xlabel("Model", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Feature", fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor='white', dpi=150)
    plt.show()
    plt.close(fig)

    LOGGER.info("Plots saved to %s", PLOTS_DIR.relative_to(PROJECT_ROOT))


def main():
    rankings_df = load_rankings(RANKING_FILES)

    if not KLRVF_CSV.exists():
        raise RuntimeError("Missing data/nfcorpus/nfcorpus_doc_klrvf.csv")
    if not CORPUS_PATH.exists():
        raise RuntimeError("Missing data/nfcorpus/nfcorpus.csv")

    feats_df = load_features(KLRVF_CSV, CORPUS_PATH)

    merged = rankings_df.merge(feats_df, on="doc_id", how="left")

    feature_columns = [
        "avg_word_length",
        "doc_length",
        "lexical_diversity",
        "readability",
        "sentence_count",
        "word_count",
    ]

    corr_df = analyze_correlations(merged, feature_columns)
    corr_df.to_csv(OUT_CORR, index=False)

    n_repeats = int(os.getenv("FRRA_N_REPEATS", "10"))
    n_jobs = int(os.getenv("FRRA_N_JOBS", "1"))
    imp_df = compute_importance(
        merged,
        feature_columns,
        target="doc_score",
        n_repeats=n_repeats,
        n_jobs=n_jobs,
    )
    imp_df.to_csv(OUT_IMPORTANCE, index=False)

    plot_correlation_heatmaps(corr_df, PLOT_CORR)
    plot_importance_heatmaps(imp_df, PLOT_IMP)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("plot_feature_analysis failed")
