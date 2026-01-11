import itertools
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rankers.registry import list_rankers

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("plot_agreement")

RANKINGS_DIR = PROJECT_ROOT / os.getenv("RANKINGS_DIR")
PLOTS_DIR = PROJECT_ROOT / os.getenv("PLOTS_DIR") / "rbo"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SPARSE = {"bm25", "tfidf"}
DENSE = {"sbert-minilm-v6", "e5-small-v2"}

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def prefix_overlap_curve(l1, l2, max_k=100):
    s1, s2 = set(), set()
    curve = []

    for d in range(1, max_k + 1):
        if d <= len(l1):
            s1.add(l1[d - 1])
        if d <= len(l2):
            s2.add(l2[d - 1])

        curve.append(len(s1 & s2) / d)

    return curve


def load_rankings(ranking_files, allowed_models):
    df = pd.concat(
        [pd.read_csv(f, dtype={"query_id": str}) for f in ranking_files],
        ignore_index=True,
    )
    df = df[df["model"].isin(allowed_models)]

    rankings = {
        (m, q): g.sort_values("doc_rank")["doc_id"].tolist()
        for (m, q), g in df.groupby(["model", "query_id"])
    }

    models = sorted(df["model"].unique())
    queries = sorted(df["query_id"].unique())

    return rankings, models, queries


def _allowed_models():
    return {m["name"] for m in list_rankers()}


def _filter_ranking_files():
    allowed = _allowed_models()
    ranking_files = []
    for path in RANKINGS_DIR.glob("rankings_*.csv"):
        model_name = path.stem.replace("rankings_", "")
        if model_name in allowed:
            ranking_files.append(path)
            continue
        LOGGER.info("Ignoring rankings for removed model: %s", model_name)
        try:
            path.unlink()
            LOGGER.info("Deleted stale rankings file: %s", path.name)
        except OSError:
            LOGGER.warning("Failed to delete stale rankings file: %s", path.name)
    return ranking_files


def plot_group(curves, title, filename, max_k):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Color palette
    colors = sns.color_palette("husl", len(curves))

    # Plot curves with confidence intervals if available
    for (label, curve), color in zip(curves.items(), colors):
        ax.plot(
            range(1, max_k + 1),
            curve,
            linewidth=2.5,
            label=label,
            color=color,
            alpha=0.9
        )

    depth_markers = [10, 50]
    region_labels = [(5, "Top\n(1-10)"), (30, "Middle\n(11-50)"), (75, "Tail\n(51-100)")]

    for depth in depth_markers:
        ax.axvline(depth, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

    # Add region labels with background
    for x_pos, text in region_labels:
        ax.text(
            x_pos, 0.98, text,
            ha="center", va="top",
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8),
            transform=ax.get_xaxis_transform()
        )

    # Styling
    ax.set_xlabel("Rank Depth (k)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Prefix Overlap", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set axis limits and ticks
    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 10, 25, 50, 75, 100])
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc='best',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10
    )

    # Add summary statistics box
    final_overlaps = [curve[-1] for curve in curves.values()]
    mean_final = np.mean(final_overlaps)
    stats_text = f"Mean overlap at k={max_k}: {mean_final:.3f}"

    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='bottom'
    )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()


def plot_combined_comparison(curves_ss, curves_sd, curves_dd, max_k):
    """Create a combined plot showing all three comparison types"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150, sharey=True)

    all_curves = [
        (curves_ss, "Sparse–Sparse", axes[0]),
        (curves_sd, "Sparse–Dense", axes[1]),
        (curves_dd, "Dense–Dense", axes[2])
    ]

    for curves, title, ax in all_curves:
        if not curves:
            continue

        colors = sns.color_palette("husl", len(curves))

        for (label, curve), color in zip(curves.items(), colors):
            ax.plot(
                range(1, max_k + 1),
                curve,
                linewidth=2,
                label=label,
                color=color,
                alpha=0.9
            )

        # Depth markers
        for depth in [10, 50]:
            ax.axvline(depth, color='gray', linestyle='--', linewidth=1, alpha=0.4)

        ax.set_xlabel("Rank Depth (k)", fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(1, max_k)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    axes[0].set_ylabel("Mean Prefix Overlap", fontsize=11, fontweight='bold')

    fig.suptitle("Ranking Agreement Across Model Types", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "agreement_combined.png", bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()


def main():
    allowed = _allowed_models()
    ranking_files = _filter_ranking_files()
    if not ranking_files:
        raise RuntimeError("No rankings_*.csv files found")

    rankings, models, queries = load_rankings(ranking_files, allowed)
    if not models:
        raise RuntimeError("No rankings for active models after filtering.")
    max_k = 100

    curves_ss = {}
    curves_sd = {}
    curves_dd = {}

    for m1, m2 in itertools.combinations(models, 2):
        per_query_curves = []

        for q in queries:
            l1 = rankings.get((m1, q), [])
            l2 = rankings.get((m2, q), [])

            if not l1 or not l2:
                continue

            per_query_curves.append(prefix_overlap_curve(l1, l2, max_k))

        if not per_query_curves:
            continue

        mean_curve = np.mean(per_query_curves, axis=0)
        label = f"{m1} vs {m2}"

        if m1 in SPARSE and m2 in SPARSE:
            curves_ss[label] = mean_curve
        elif m1 in DENSE and m2 in DENSE:
            curves_dd[label] = mean_curve
        else:
            curves_sd[label] = mean_curve

    # Individual plots
    if curves_ss:
        plot_group(
            curves_ss,
            title="Sparse–Sparse Agreement Across Ranking Depths",
            filename="agreement_sparse_sparse.png",
            max_k=max_k,
        )

    if curves_sd:
        plot_group(
            curves_sd,
            title="Sparse–Dense Agreement Across Ranking Depths",
            filename="agreement_sparse_dense.png",
            max_k=max_k,
        )

    if curves_dd:
        plot_group(
            curves_dd,
            title="Dense–Dense Agreement Across Ranking Depths",
            filename="agreement_dense_dense.png",
            max_k=max_k,
        )

    # Combined comparison plot
    plot_combined_comparison(curves_ss, curves_sd, curves_dd, max_k)

    LOGGER.info("Plots saved to %s", PLOTS_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("plot_agreement failed")
