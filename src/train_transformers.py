"""
train_transformers.py — GNN + Transformer, neighbour-constrained prediction.

Architecture
────────────
1. GraphSAGE encodes every node into a 64-dim spatial embedding.
2. Trajectory windows are embedded via GNN node lookups → [B, T, D].
3. Transformer encoder reads the window with a causal mask → context vector.
4. Context vector is dot-product scored against each candidate neighbour's
   GNN embedding → logits over local candidates (max_degree options).
5. Both GNN and Transformer are trained end-to-end.

This is directly comparable to train_gnn.py: same task, same dataset,
same scoring head.  The only difference is the GNN uses only the last node
as query while the Transformer uses the full trajectory window.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import load_graph, build_graph_data, generate_trajectories, train_val_test_split
from src.dataset import build_neighbour_dataset
from src.graph_model import GraphSAGEModel
from src.transformer_model import TransformerModel


# ── Hyperparameters ───────────────────────────────────────────────────────────

PLACE       = "Cottbus, Germany"
NUM_TRAJ    = 5000
WALK_LENGTH = 8
SEQ_LEN     = 5
MAX_DEGREE  = 8
HIDDEN_DIM  = 64
EPOCHS      = 20
LR          = 0.001
CLIP_NORM   = 1.0
RESULTS_DIR = "results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def topk_accuracy(logits, targets, k):
    k = min(k, logits.size(1))
    _, top_idx = torch.topk(logits, k, dim=1)
    return top_idx.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()


def random_baseline(valid_masks):
    degrees = valid_masks.sum(dim=1).float()
    return (1.0 / degrees).mean().item()


def make_causal_mask(seq_len, device):
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


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

    causal_mask = make_causal_mask(SEQ_LEN, device)

    # ── Models ────────────────────────────────────────────────────────────────
    gnn = GraphSAGEModel(input_dim, HIDDEN_DIM, num_nodes, MAX_DEGREE).to(device)
    transformer = TransformerModel(
        num_nodes  = num_nodes,
        embed_dim  = HIDDEN_DIM,
        max_degree = MAX_DEGREE,
    ).to(device)

    optimizer = optim.Adam(
        list(gnn.parameters()) + list(transformer.parameters()), lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    rows = [["epoch", "train_loss", "train_top1", "train_top3",
             "val_top1", "val_top3", "random_baseline"]]

    for epoch in range(1, EPOCHS + 1):
        gnn.train()
        transformer.train()

        # GNN: spatial embeddings for all nodes [N, D]
        node_emb = gnn.forward_embeddings(x, edge_index)

        # Embed trajectory windows via node lookup [B, T, D]
        embedded = node_emb[tr_seqs]

        # Transformer: context vector → candidate scores [B, max_degree]
        logits = transformer(embedded, node_emb, tr_cands, tr_masks, mask=causal_mask)
        loss   = criterion(logits, tr_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(gnn.parameters()) + list(transformer.parameters()),
            max_norm=CLIP_NORM,
        )
        optimizer.step()

        tr_top1 = topk_accuracy(logits.detach(), tr_targets, k=1)
        tr_top3 = topk_accuracy(logits.detach(), tr_targets, k=3)

        # ── Validation ────────────────────────────────────────────────────────
        gnn.eval()
        transformer.eval()
        with torch.no_grad():
            va_node_emb = gnn.forward_embeddings(x, edge_index)
            va_embedded = va_node_emb[va_seqs]
            va_logits   = transformer(va_embedded, va_node_emb, va_cands, va_masks, mask=causal_mask)
            va_top1     = topk_accuracy(va_logits, va_targets, k=1)
            va_top3     = topk_accuracy(va_logits, va_targets, k=3)

        print(
            f"Epoch {epoch:>2}/{EPOCHS} | Loss: {loss.item():.4f} | "
            f"Train Top-1: {tr_top1:.4f}  Top-3: {tr_top3:.4f} | "
            f"Val Top-1: {va_top1:.4f}  Top-3: {va_top3:.4f} | "
            f"Random: {rand_base:.4f}"
        )
        rows.append([epoch, loss.item(), tr_top1, tr_top3, va_top1, va_top3, rand_base])

    csv_path = os.path.join(RESULTS_DIR, "transformer_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nMetrics saved to {csv_path}")


if __name__ == "__main__":
    train()