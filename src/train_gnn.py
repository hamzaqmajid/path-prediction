"""
train_gnn.py — GraphSAGE baseline, neighbour-constrained next-node prediction.

Changes in this version
────────────────────────
1. data_loader now generates direction-biased walks (alpha=3.0) and 7-dim
   node features — input_dim updated automatically via x.size(1).
2. 50 epochs instead of 20 — previous runs had not converged.
3. ReduceLROnPlateau scheduler — halves LR when val Top-1 stops improving,
   prevents the loss spikes seen in the previous GNN run.
4. NUM_TRAJ increased to 10 000 — more samples now that the walks carry
   real directional signal worth learning from.
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


# ── Hyperparameters ───────────────────────────────────────────────────────────

PLACE       = "Cottbus, Germany"
NUM_TRAJ    = 10000
WALK_LENGTH = 10        # longer walks → more (seq, target) pairs per traj
WALK_ALPHA  = 3.0       # direction bias; 0 = uniform (old behaviour)
SEQ_LEN     = 5
MAX_DEGREE  = 8
HIDDEN_DIM  = 64
EPOCHS      = 50
LR          = 0.001
RESULTS_DIR = "results"


# ── Metrics ───────────────────────────────────────────────────────────────────

def topk_accuracy(logits, targets, k):
    k = min(k, logits.size(1))
    _, top_idx = torch.topk(logits, k, dim=1)
    return top_idx.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()


def random_baseline(valid_masks):
    return (1.0 / valid_masks.sum(dim=1).float()).mean().item()


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cpu")

    # ── Graph ─────────────────────────────────────────────────────────────────
    print("Loading graph ...")
    G = load_graph(PLACE)
    x, edge_index, node_map = build_graph_data(G)

    num_nodes = x.size(0)
    input_dim = x.size(1)   # 7 with new features
    print(f"  Nodes: {num_nodes}  |  Input dim: {input_dim}")

    x          = x.to(device)
    edge_index = edge_index.to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Generating {NUM_TRAJ} direction-biased trajectories (alpha={WALK_ALPHA}) ...")
    all_traj = generate_trajectories(
        G, node_map, num_traj=NUM_TRAJ, length=WALK_LENGTH, alpha=WALK_ALPHA
    )
    train_traj, val_traj, _ = train_val_test_split(all_traj)

    print("Building neighbour datasets ...")
    tr_seqs, tr_cands, tr_targets, tr_masks = build_neighbour_dataset(
        train_traj, G, node_map, seq_len=SEQ_LEN, max_degree=MAX_DEGREE)
    va_seqs, va_cands, va_targets, va_masks = build_neighbour_dataset(
        val_traj,   G, node_map, seq_len=SEQ_LEN, max_degree=MAX_DEGREE)

    for t in [tr_seqs, tr_cands, tr_targets, tr_masks]:
        t = t.to(device)
    tr_seqs    = tr_seqs.to(device);    tr_cands   = tr_cands.to(device)
    tr_targets = tr_targets.to(device); tr_masks   = tr_masks.to(device)
    va_seqs    = va_seqs.to(device);    va_cands   = va_cands.to(device)
    va_targets = va_targets.to(device); va_masks   = va_masks.to(device)

    rand_base = random_baseline(tr_masks)
    print(f"  Train samples         : {len(tr_seqs)}")
    print(f"  Val   samples         : {len(va_seqs)}")
    print(f"  Random Top-1 baseline : {rand_base:.4f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = GraphSAGEModel(input_dim, HIDDEN_DIM, num_nodes, MAX_DEGREE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Halve LR when val Top-1 stops improving for 5 consecutive epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    rows = [["epoch", "lr", "train_loss", "train_top1", "train_top3",
             "val_top1", "val_top3", "random_baseline"]]

    best_val_top1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        logits = model(x, edge_index, tr_seqs, tr_cands, tr_masks)
        loss   = criterion(logits, tr_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tr_top1 = topk_accuracy(logits.detach(), tr_targets, k=1)
        tr_top3 = topk_accuracy(logits.detach(), tr_targets, k=3)

        model.eval()
        with torch.no_grad():
            va_logits = model(x, edge_index, va_seqs, va_cands, va_masks)
            va_top1   = topk_accuracy(va_logits, va_targets, k=1)
            va_top3   = topk_accuracy(va_logits, va_targets, k=3)

        scheduler.step(va_top1)
        current_lr = optimizer.param_groups[0]['lr']

        if va_top1 > best_val_top1:
            best_val_top1 = va_top1
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "gnn_best.pt"))

        print(
            f"Epoch {epoch:>2}/{EPOCHS} | lr: {current_lr:.5f} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Top-1: {tr_top1:.4f}  Top-3: {tr_top3:.4f} | "
            f"Val Top-1: {va_top1:.4f}  Top-3: {va_top3:.4f} | "
            f"Random: {rand_base:.4f}"
        )
        rows.append([epoch, current_lr, loss.item(),
                     tr_top1, tr_top3, va_top1, va_top3, rand_base])

    print(f"\nBest val Top-1: {best_val_top1:.4f}")

    csv_path = os.path.join(RESULTS_DIR, "gnn_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Metrics saved to {csv_path}")


if __name__ == "__main__":
    train()