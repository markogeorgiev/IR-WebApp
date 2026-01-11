from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from compute_features import analyze_text
except Exception:  # pragma: no cover - fallback for module path differences
    from scripts.compute_features import analyze_text

from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("plot_klrvf_comparison")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()

CORPUS_PATH = PROJECT_ROOT / os.getenv("CORPUS_PATH")
MODIFIED_DIR = PROJECT_ROOT / os.getenv("MODIFIED_DIR")
PLOTS_DIR = PROJECT_ROOT / os.getenv("PLOTS_DIR") / "klrvf_comparisons"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "token_count",
    "sentence_count",
    "avg_sentence_length",
    "avg_word_length",
    "type_token_ratio",
    "flesch_reading_ease",
]

METRIC_LABELS = {
    "token_count": "Token Count",
    "sentence_count": "Sentence Count",
    "avg_sentence_length": "Avg Sentence Length",
    "avg_word_length": "Avg Word Length",
    "type_token_ratio": "Type-Token Ratio",
    "flesch_reading_ease": "Flesch Reading Ease",
}


def load_nfcorpus():
    df = pd.read_csv(CORPUS_PATH)
    df["full_text"] = (
        df["title"].fillna("") + "\n" + df["abstract"].fillna("")
    ).str.strip()
    return df.set_index("doc_id")


def load_modified(path: Path):
    df = pd.read_csv(path)
    df["full_text"] = (
        df["title"].fillna("") + "\n" + df["abstract"].fillna("")
    ).str.strip()
    return df


def compute_features(df, version):
    rows = []
    for _, row in df.iterrows():
        feats = analyze_text(row["full_text"])
        feats["doc_id"] = row["doc_id"]
        feats["version"] = version
        rows.append(feats)
    return pd.DataFrame(rows)


def calculate_global_limits(feat_df):
    limits = {}
    for metric in METRICS:
        min_val = feat_df[metric].min()
        max_val = feat_df[metric].max()
        range_val = max_val - min_val
        padding = range_val * 0.1 if range_val > 0 else 1
        limits[metric] = (min_val - padding, max_val + padding)
    return limits


def plot_comprehensive_comparison(feat_df, doc_id, global_limits, model_name):
    sub = feat_df[feat_df["doc_id"] == doc_id].set_index("version")

    if len(sub) < 2:
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(
        3, 3, hspace=0.3, wspace=0.3, top=0.93, bottom=0.07, left=0.07, right=0.97
    )

    colors = ["#2E86AB", "#A23B72"]
    delta_color_pos = "#06A77D"
    delta_color_neg = "#D62839"

    for idx, metric in enumerate(METRICS):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        values = sub[metric].values
        x_pos = np.arange(len(sub))

        bars = ax.bar(
            x_pos,
            values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Original", "Modified"], fontsize=11, fontweight="bold")
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(global_limits[metric])
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_delta = fig.add_subplot(gs[2, :])
    delta = sub.loc["modified", METRICS] - sub.loc["original", METRICS]

    bar_colors = [delta_color_pos if d > 0 else delta_color_neg for d in delta.values]

    x_pos = np.arange(len(METRICS))
    bars = ax_delta.bar(
        x_pos,
        delta.values,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar, val in zip(bars, delta.values):
        height = bar.get_height()
        y_pos = height if height > 0 else height
        va = "bottom" if height > 0 else "top"
        ax_delta.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{val:+.2f}",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    ax_delta.axhline(0, color="black", linestyle="-", linewidth=2, alpha=0.7)
    ax_delta.set_xticks(x_pos)
    ax_delta.set_xticklabels(
        [METRIC_LABELS[m] for m in METRICS],
        rotation=45,
        ha="right",
        fontsize=10,
        fontweight="bold",
    )
    ax_delta.set_ylabel("Change (Modified - Original)", fontsize=12, fontweight="bold")
    ax_delta.set_title(
        "Metric Changes: Modified vs Original", fontsize=13, fontweight="bold", pad=15
    )
    ax_delta.grid(axis="y", alpha=0.3, linestyle="--")
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)

    fig.suptitle(
        f"KLRVF Analysis: Document {doc_id}", fontsize=16, fontweight="bold", y=0.98
    )

    plt.savefig(
        PLOTS_DIR / f"{doc_id}_{model_name}_comprehensive.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    nf = load_nfcorpus()

    all_features = []

    for edited_csv in MODIFIED_DIR.glob("edited_*.csv"):
        mod = load_modified(edited_csv)

        model_name = edited_csv.stem.replace("edited_", "")

        orig_rows = []
        for _, row in mod.iterrows():
            doc_id = row["doc_id"]
            if doc_id not in nf.index:
                continue
            orig_rows.append({"doc_id": doc_id, "full_text": nf.loc[doc_id, "full_text"]})

        if len(orig_rows) == 0:
            continue

        orig_df = pd.DataFrame(orig_rows)

        orig_feats = compute_features(orig_df, "original")
        mod_feats = compute_features(mod, "modified")

        feat_df = pd.concat([orig_feats, mod_feats], ignore_index=True)
        all_features.append((feat_df, model_name))

    if not all_features:
        LOGGER.warning("No data found to plot")
        return

    combined_features = pd.concat(
        [feat_df for feat_df, model_name in all_features], ignore_index=True
    )
    global_limits = calculate_global_limits(combined_features)

    for feat_df, model_name in all_features:
        for doc_id in feat_df["doc_id"].unique():
            plot_comprehensive_comparison(feat_df, doc_id, global_limits, model_name)

    LOGGER.info(
        "KLRVF comparison plots saved to %s", PLOTS_DIR.relative_to(PROJECT_ROOT)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("plot_klrvf_comparison failed")
