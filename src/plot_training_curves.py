"""
plot_training_curves.py — Training curve figure for thesis.

Reads results/gnn_metrics.csv and results/transformer_metrics.csv and
produces a single publication-quality figure with:
  - Val Top-1 vs epoch for both models
  - Random baseline as a horizontal dashed line
  - Saved to results/training_curves.png (300 dpi)

Run from project root:
    python src/plot_training_curves.py
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


RESULTS_DIR = "results"
OUT_PATH    = os.path.join(RESULTS_DIR, "training_curves.png")


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
    return rows


def main():
    gnn_path = os.path.join(RESULTS_DIR, "gnn_metrics.csv")
    tfm_path = os.path.join(RESULTS_DIR, "transformer_metrics.csv")

    for p in [gnn_path, tfm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run train_gnn.py and train_transformers.py first."
            )

    gnn_rows = load_csv(gnn_path)
    tfm_rows = load_csv(tfm_path)

    gnn_epochs    = [int(r["epoch"])       for r in gnn_rows]
    gnn_val_top1  = [float(r["val_top1"])  for r in gnn_rows]
    gnn_val_top3  = [float(r["val_top3"])  for r in gnn_rows]
    gnn_random    = float(gnn_rows[0]["random_baseline"])

    tfm_epochs    = [int(r["epoch"])       for r in tfm_rows]
    tfm_val_top1  = [float(r["val_top1"])  for r in tfm_rows]
    tfm_val_top3  = [float(r["val_top3"])  for r in tfm_rows]
    tfm_random    = float(tfm_rows[0]["random_baseline"])

    random_baseline = (gnn_random + tfm_random) / 2

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "GNN vs Transformer — Validation Accuracy\n"
        "Neighbour-constrained next-node prediction · Cottbus road network",
        fontsize=13, fontweight='bold', y=1.01
    )

    BLUE   = "#2563EB"
    ORANGE = "#EA580C"
    GREY   = "#6B7280"

    # ── Top-1 ─────────────────────────────────────────────────────────────────
    ax1.plot(gnn_epochs, gnn_val_top1,
             color=BLUE,   linewidth=2,   label="GNN (GraphSAGE)")
    ax1.plot(tfm_epochs, tfm_val_top1,
             color=ORANGE, linewidth=2,   label="GNN + Transformer")
    ax1.axhline(random_baseline, color=GREY, linewidth=1.2,
                linestyle='--', label=f"Random baseline ({random_baseline:.3f})")

    # Annotate best values
    best_gnn = max(gnn_val_top1)
    best_tfm = max(tfm_val_top1)
    ax1.annotate(f"Best: {best_gnn:.4f}",
                 xy=(gnn_epochs[gnn_val_top1.index(best_gnn)], best_gnn),
                 xytext=(5, -18), textcoords='offset points',
                 color=BLUE, fontsize=9)
    ax1.annotate(f"Best: {best_tfm:.4f}",
                 xy=(tfm_epochs[tfm_val_top1.index(best_tfm)], best_tfm),
                 xytext=(5, 6), textcoords='offset points',
                 color=ORANGE, fontsize=9)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Top-1 Accuracy", fontsize=11)
    ax1.set_title("Top-1 Accuracy", fontsize=11)
    ax1.set_ylim(0.30, min(1.0, max(best_tfm, best_gnn) + 0.08))
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Top-3 ─────────────────────────────────────────────────────────────────
    ax2.plot(gnn_epochs, gnn_val_top3,
             color=BLUE,   linewidth=2,   label="GNN (GraphSAGE)")
    ax2.plot(tfm_epochs, tfm_val_top3,
             color=ORANGE, linewidth=2,   label="GNN + Transformer")

    best_gnn3 = max(gnn_val_top3)
    best_tfm3 = max(tfm_val_top3)
    ax2.annotate(f"Best: {best_gnn3:.4f}",
                 xy=(gnn_epochs[gnn_val_top3.index(best_gnn3)], best_gnn3),
                 xytext=(5, -14), textcoords='offset points',
                 color=BLUE, fontsize=9)
    ax2.annotate(f"Best: {best_tfm3:.4f}",
                 xy=(tfm_epochs[tfm_val_top3.index(best_tfm3)], best_tfm3),
                 xytext=(5, 4), textcoords='offset points',
                 color=ORANGE, fontsize=9)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Top-3 Accuracy", fontsize=11)
    ax2.set_title("Top-3 Accuracy", fontsize=11)
    ax2.set_ylim(0.90, 1.01)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
