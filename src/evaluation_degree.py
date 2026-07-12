"""
evaluate_by_degree.py — Degree-stratified evaluation.

Loads the best saved checkpoints (results/gnn_best.pt and
results/transformer_best.pt) and evaluates Top-1 accuracy broken down
by the degree of the current node (the node whose next step is predicted).

At degree-2 nodes (dead-ends or through-roads) the model has at most one
real choice after anti-backtracking, so accuracy is trivially high.
At degree-3 and degree-4+ nodes the problem is genuinely hard and the
Transformer's history advantage is most visible.

Output
──────
- Printed table: degree → (GNN Top-1, Transformer Top-1, delta, sample count)
- results/degree_stratified.csv

Run from project root:
    python src/evaluate_by_degree.py
"""

import os
import csv
from collections import defaultdict

import torch

from src.data_loader import (
    load_graph, build_graph_data,
    generate_trajectories, train_val_test_split,
)
from src.dataset import build_neighbour_dataset
from src.graph_model import GraphSAGEModel
from src.transformer_model import TransformerModel


# ── Config ────────────────────────────────────────────────────────────────────

PLACE       = "Cottbus, Germany"
NUM_TRAJ    = 10_000
WALK_LENGTH = 10
WALK_ALPHA  = 3.0
SEQ_LEN     = 5
MAX_DEGREE  = 8
HIDDEN_DIM  = 64
RESULTS_DIR = "results"
GNN_CKPT    = os.path.join(RESULTS_DIR, "gnn_best.pt")
TFM_CKPT    = os.path.join(RESULTS_DIR, "transformer_best.pt")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_causal_mask(seq_len, device):
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


def node_degree_map(G, node_map):
    """Return dict: contiguous_index → degree in G."""
    return {node_map[n]: G.degree[n] for n in G.nodes()}


def evaluate_by_degree(logits, targets, seqs, degree_map):
    """
    Returns dict: degree → {"correct": int, "total": int}
    The degree used is the degree of the last node in each input sequence
    (the node whose next step is being predicted).
    """
    preds   = torch.argmax(logits, dim=1)           # [B]
    correct = (preds == targets)                    # [B] bool

    buckets = defaultdict(lambda: {"correct": 0, "total": 0})

    for i in range(len(targets)):
        current_node = seqs[i, -1].item()           # last node in window
        deg          = degree_map.get(current_node, 0)
        bucket       = deg if deg <= 4 else 5       # 5 = "4+"
        buckets[bucket]["total"]   += 1
        buckets[bucket]["correct"] += int(correct[i].item())

    return buckets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")

    # ── Check checkpoints exist ───────────────────────────────────────────────
    for ckpt in [GNN_CKPT, TFM_CKPT]:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"{ckpt} not found.\n"
                "Run train_gnn.py and train_transformers.py first to generate checkpoints."
            )

    # ── Graph + data ──────────────────────────────────────────────────────────
    print("Loading graph ...")
    G = load_graph(PLACE)
    x, edge_index, node_map = build_graph_data(G)
    x          = x.to(device)
    edge_index = edge_index.to(device)

    print(f"Generating {NUM_TRAJ} trajectories ...")
    all_traj = generate_trajectories(
        G, node_map, num_traj=NUM_TRAJ, length=WALK_LENGTH, alpha=WALK_ALPHA
    )
    _, val_traj, _ = train_val_test_split(all_traj)

    print("Building val dataset ...")
    va_seqs, va_cands, va_targets, va_masks = build_neighbour_dataset(
        val_traj, G, node_map, seq_len=SEQ_LEN, max_degree=MAX_DEGREE
    )
    va_seqs    = va_seqs.to(device);    va_cands   = va_cands.to(device)
    va_targets = va_targets.to(device); va_masks   = va_masks.to(device)

    deg_map     = node_degree_map(G, node_map)
    causal_mask = make_causal_mask(SEQ_LEN, device)

    num_nodes = x.size(0)
    input_dim = x.size(1)

    # ── Load GNN ──────────────────────────────────────────────────────────────
    print("Loading GNN checkpoint ...")
    gnn_eval = GraphSAGEModel(input_dim, HIDDEN_DIM, num_nodes, MAX_DEGREE).to(device)
    gnn_eval.load_state_dict(torch.load(GNN_CKPT, map_location=device))
    gnn_eval.eval()

    with torch.no_grad():
        gnn_logits = gnn_eval(x, edge_index, va_seqs, va_cands, va_masks)

    gnn_buckets = evaluate_by_degree(gnn_logits, va_targets, va_seqs, deg_map)

    # ── Load Transformer ──────────────────────────────────────────────────────
    print("Loading Transformer checkpoint ...")
    gnn_tfm = GraphSAGEModel(input_dim, HIDDEN_DIM, num_nodes, MAX_DEGREE).to(device)
    tfm     = TransformerModel(
        num_nodes=num_nodes, embed_dim=HIDDEN_DIM, max_degree=MAX_DEGREE
    ).to(device)

    ckpt = torch.load(TFM_CKPT, map_location=device)
    gnn_tfm.load_state_dict(ckpt["gnn"])
    tfm.load_state_dict(ckpt["transformer"])
    gnn_tfm.eval(); tfm.eval()

    with torch.no_grad():
        node_emb    = gnn_tfm.forward_embeddings(x, edge_index)
        embedded    = node_emb[va_seqs]
        tfm_logits  = tfm(embedded, node_emb, va_cands, va_masks, mask=causal_mask)

    tfm_buckets = evaluate_by_degree(tfm_logits, va_targets, va_seqs, deg_map)

    # ── Build and print table ─────────────────────────────────────────────────
    degree_labels = {2: "2", 3: "3", 4: "4", 5: "4+"}
    all_degrees   = sorted(set(list(gnn_buckets.keys()) + list(tfm_buckets.keys())))

    rows = []
    print("\n" + "═" * 72)
    print(f"{'Degree':>8} {'Samples':>9} {'GNN Top-1':>11} "
          f"{'TFM Top-1':>11} {'Delta':>8} {'% of val':>9}")
    print("─" * 72)

    total_samples = sum(v["total"] for v in gnn_buckets.values())

    for deg in all_degrees:
        g = gnn_buckets.get(deg, {"correct": 0, "total": 0})
        t = tfm_buckets.get(deg, {"correct": 0, "total": 0})

        n          = g["total"]
        gnn_acc    = g["correct"] / n if n > 0 else 0.0
        tfm_acc    = t["correct"] / n if n > 0 else 0.0
        delta      = tfm_acc - gnn_acc
        pct_of_val = 100 * n / total_samples if total_samples > 0 else 0.0
        label      = degree_labels.get(deg, str(deg))

        print(f"{'deg ' + label:>8} {n:>9,} {gnn_acc:>11.4f} "
              f"{tfm_acc:>11.4f} {delta:>+8.4f} {pct_of_val:>8.1f}%")

        rows.append({
            "degree":     label,
            "samples":    n,
            "pct_of_val": round(pct_of_val, 2),
            "gnn_top1":   round(gnn_acc, 4),
            "tfm_top1":   round(tfm_acc, 4),
            "delta":      round(delta, 4),
        })

    print("═" * 72)

    # ── Overall ───────────────────────────────────────────────────────────────
    gnn_overall = sum(v["correct"] for v in gnn_buckets.values()) / total_samples
    tfm_overall = sum(v["correct"] for v in tfm_buckets.values()) / total_samples
    print(f"{'Overall':>8} {total_samples:>9,} {gnn_overall:>11.4f} "
          f"{tfm_overall:>11.4f} {tfm_overall - gnn_overall:>+8.4f} {'100.0%':>9}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "degree_stratified.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()