"""
ablation_seqlen.py — Sequence length ablation study.

Tests the Transformer at seq_len ∈ {1, 3, 5, 7} with everything else fixed.
seq_len=1 is the fairest comparison to the GNN (no history), so the delta
from seq_len=1 → seq_len=5 isolates the value of trajectory history.

Results saved to results/ablation_seqlen.csv and printed as a table.

"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import (
    load_graph, build_graph_data,
    generate_trajectories, train_val_test_split,
)
from src.dataset import build_neighbour_dataset
from src.graph_model import GraphSAGEModel
from src.transformer_model import TransformerModel


# ── Fixed hyperparameters ─────────────────────────────────────────────────────

PLACE       = "Cottbus, Germany"
NUM_TRAJ    = 10_000
WALK_LENGTH = 10
MAX_DEGREE  = 8
HIDDEN_DIM  = 64
EPOCHS      = 50
LR          = 0.001
CLIP_NORM   = 1.0
RESULTS_DIR = "results"

# Ablation grid
SEQ_LENS    = [1, 3, 5, 7]
ALPHAS      = [3.0]          # add 0.0 here to also test uniform-walk variant


# ── Helpers ───────────────────────────────────────────────────────────────────

def topk_accuracy(logits, targets, k):
    k = min(k, logits.size(1))
    _, top_idx = torch.topk(logits, k, dim=1)
    return top_idx.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()


def random_baseline(valid_masks):
    return (1.0 / valid_masks.sum(dim=1).float()).mean().item()


def make_causal_mask(seq_len, device):
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


def run_variant(G, x, edge_index, node_map, device,
                seq_len, alpha, num_traj=NUM_TRAJ):
    """Train one (seq_len, alpha) variant and return best val Top-1."""

    num_nodes = x.size(0)
    input_dim = x.size(1)

    all_traj = generate_trajectories(
        G, node_map, num_traj=num_traj, length=WALK_LENGTH, alpha=alpha
    )
    train_traj, val_traj, _ = train_val_test_split(all_traj)

    tr_seqs, tr_cands, tr_targets, tr_masks = build_neighbour_dataset(
        train_traj, G, node_map, seq_len=seq_len, max_degree=MAX_DEGREE)
    va_seqs, va_cands, va_targets, va_masks = build_neighbour_dataset(
        val_traj,   G, node_map, seq_len=seq_len, max_degree=MAX_DEGREE)

    if len(tr_seqs) == 0 or len(va_seqs) == 0:
        print(f"  [skip] seq_len={seq_len} alpha={alpha} — no samples generated")
        return None

    tr_seqs    = tr_seqs.to(device);    tr_cands   = tr_cands.to(device)
    tr_targets = tr_targets.to(device); tr_masks   = tr_masks.to(device)
    va_seqs    = va_seqs.to(device);    va_cands   = va_cands.to(device)
    va_targets = va_targets.to(device); va_masks   = va_masks.to(device)

    rand_base   = random_baseline(tr_masks)
    causal_mask = make_causal_mask(seq_len, device)

    gnn = GraphSAGEModel(input_dim, HIDDEN_DIM, num_nodes, MAX_DEGREE).to(device)
    transformer = TransformerModel(
        num_nodes=num_nodes, embed_dim=HIDDEN_DIM, max_degree=MAX_DEGREE
    ).to(device)

    all_params = list(gnn.parameters()) + list(transformer.parameters())
    optimizer  = optim.Adam(all_params, lr=LR)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion  = nn.CrossEntropyLoss()

    best_val_top1 = 0.0
    best_val_top3 = 0.0

    for epoch in range(1, EPOCHS + 1):
        gnn.train(); transformer.train()

        node_emb = gnn.forward_embeddings(x, edge_index)
        embedded = node_emb[tr_seqs]
        logits   = transformer(embedded, node_emb, tr_cands, tr_masks, mask=causal_mask)
        loss     = criterion(logits, tr_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, CLIP_NORM)
        optimizer.step()

        gnn.eval(); transformer.eval()
        with torch.no_grad():
            va_node_emb = gnn.forward_embeddings(x, edge_index)
            va_embedded = va_node_emb[va_seqs]
            va_logits   = transformer(va_embedded, va_node_emb,
                                      va_cands, va_masks, mask=causal_mask)
            va_top1 = topk_accuracy(va_logits, va_targets, k=1)
            va_top3 = topk_accuracy(va_logits, va_targets, k=3)

        scheduler.step(va_top1)

        if va_top1 > best_val_top1:
            best_val_top1 = va_top1
            best_val_top3 = va_top3

        if epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    ep {epoch:>2} | lr {lr_now:.5f} | "
                  f"val Top-1 {va_top1:.4f} | best {best_val_top1:.4f}")

    return {
        "seq_len":       seq_len,
        "alpha":         alpha,
        "train_samples": len(tr_seqs),
        "val_samples":   len(va_seqs),
        "random_base":   round(rand_base, 4),
        "best_val_top1": round(best_val_top1, 4),
        "best_val_top3": round(best_val_top3, 4),
        "gain_over_random": round(best_val_top1 - rand_base, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cpu")

    print("Loading graph ...")
    G = load_graph(PLACE)
    x, edge_index, node_map = build_graph_data(G)
    x          = x.to(device)
    edge_index = edge_index.to(device)
    print(f"  Nodes: {x.size(0)}  |  Input dim: {x.size(1)}")

    results = []

    for alpha in ALPHAS:
        for seq_len in SEQ_LENS:
            label = f"seq_len={seq_len}, alpha={alpha}"
            print(f"\n── Variant: {label} ──")
            row = run_variant(G, x, edge_index, node_map, device,
                              seq_len=seq_len, alpha=alpha)
            if row:
                results.append(row)
                print(f"  → Best val Top-1: {row['best_val_top1']:.4f}  "
                      f"(+{row['gain_over_random']:.4f} over random)")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print(f"{'seq_len':>8} {'alpha':>6} {'random':>8} "
          f"{'Top-1':>8} {'Top-3':>8} {'gain':>8}")
    print("─" * 72)
    for r in results:
        print(f"{r['seq_len']:>8} {r['alpha']:>6} {r['random_base']:>8.4f} "
              f"{r['best_val_top1']:>8.4f} {r['best_val_top3']:>8.4f} "
              f"{r['gain_over_random']:>+8.4f}")
    print("═" * 72)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "ablation_seqlen.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
