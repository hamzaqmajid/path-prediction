"""
train_gnn.py — GraphSAGE baseline, neighbour-constrained next-node prediction.

Task
────
Given the last node in a trajectory window, predict which of its road-network
neighbours is visited next.  Output space = local degree (2–4 options), not
all 2 128 nodes.  Random baseline Top-1 ≈ 30–40 %.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import load_graph, build_graph_data, generate_trajectories, train_val_test_split
from src.dataset import build_neighbour_dataset
from src.graph_model import GraphSAGEModel


# ── Hyperparameters ───────────────────────────────────────────────────────────

PLACE       = "Cottbus, Germany"
NUM_TRAJ    = 5000
WALK_LENGTH = 8
SEQ_LEN     = 5
MAX_DEGREE  = 8
HIDDEN_DIM  = 64
EPOCHS      = 20
LR          = 0.001
RESULTS_DIR = "results"


# ── Metrics ───────────────────────────────────────────────────────────────────

def topk_accuracy(logits, targets, k):
    """Top-k accuracy over local candidate list (max_degree classes)."""
    k = min(k, logits.size(1))
    _, top_idx = torch.topk(logits, k, dim=1)
    correct = top_idx.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def random_baseline(valid_masks):
    """Expected Top-1 accuracy of a uniform random policy."""
    degrees = valid_masks.sum(dim=1).float()
    return (1.0 / degrees).mean().item()


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cpu")

    # ── Graph ─────────────────────────────────────────────────────────────────
    print("Loading graph ...")
    G = load_graph(PLACE)
    x, edge_index, node_map = build_graph_data(G)

    num_nodes = x.size(0)
    input_dim = x.size(1)
    print(f"  Nodes: {num_nodes}  |  Input dim: {input_dim}")

    x          = x.to(device)
    edge_index = edge_index.to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Generating trajectories ...")
    all_traj = generate_trajectories(G, node_map, num_traj=NUM_TRAJ, length=WALK_LENGTH)
    train_traj, val_traj, _ = train_val_test_split(all_traj)

    print("Building neighbour datasets ...")
    tr_seqs, tr_cands, tr_targets, tr_masks = build_neighbour_dataset(
        train_traj, G, node_map, seq_len=SEQ_LEN, max_degree=MAX_DEGREE)
    va_seqs, va_cands, va_targets, va_masks = build_neighbour_dataset(
        val_traj,   G, node_map, seq_len=SEQ_LEN, max_degree=MAX_DEGREE)

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
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    rows = [["epoch", "train_loss", "train_top1", "train_top3",
             "val_top1", "val_top3", "random_baseline"]]

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

        print(
            f"Epoch {epoch:>2}/{EPOCHS} | Loss: {loss.item():.4f} | "
            f"Train Top-1: {tr_top1:.4f}  Top-3: {tr_top3:.4f} | "
            f"Val Top-1: {va_top1:.4f}  Top-3: {va_top3:.4f} | "
            f"Random: {rand_base:.4f}"
        )
        rows.append([epoch, loss.item(), tr_top1, tr_top3, va_top1, va_top3, rand_base])

    csv_path = os.path.join(RESULTS_DIR, "gnn_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nMetrics saved to {csv_path}")


if __name__ == "__main__":
    train()