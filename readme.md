# GNN vs Transformer for Road Network Path Prediction

## Project Overview

This project compares **Graph Neural Networks (GNNs)** and **Transformer models** for next-node prediction on a real road network. Given a partial trajectory of visited intersections, the goal is to predict which road the vehicle takes next.

The project is designed as a portfolio piece for research in AI-driven map matching and path prediction — directly relevant to autonomous driving and navigation systems.

---

## Problem Statement

**Input:** A sequence of visited road intersections (nodes) on the Cottbus, Germany road network extracted from OpenStreetMap.

**Output:** Which neighbour of the current node is visited next.

**Why neighbour-constrained output?**
At each intersection, a vehicle can only proceed to one of 2–4 directly connected roads. Predicting the next node out of all ~2,000 nodes in the graph is both unlearnable from random-walk data and architecturally wrong — a map-matching system never considers roads that are not physically reachable from the current position. The output space is therefore constrained to the local neighbourhood, making this a realistic and learnable formulation.

---

## Modeling Approach

### GNN Baseline (GraphSAGE)

- Encodes every road node into a 64-dimensional spatial embedding using two GraphSAGE layers
- Uses only the **current node** (last visited intersection) as the query
- Scores each candidate neighbour via dot-product with the query embedding
- No trajectory history — purely spatial, memoryless

This establishes the ceiling for what spatial structure alone can achieve.

### Transformer Model (GNN + Transformer, end-to-end)

- The same GraphSAGE encoder produces node embeddings
- A **5-node trajectory window** is embedded via GNN node lookups and fed to a Transformer encoder with a causal attention mask
- The last-token context vector is dot-product scored against each candidate neighbour
- Both GNN and Transformer are trained jointly end-to-end

The Transformer's advantage over the GNN baseline is purely temporal: it uses trajectory history to resolve directional ambiguity at intersections.

### Shared scoring head (pointer network)

Both models use the same output mechanism:

```
query_vector · candidate_embedding  →  score per candidate
softmax over valid candidates only  →  probability distribution
```

This isolates the architectural variable (spatial-only vs spatial+temporal) and makes the comparison fair.

---

## Architecture Diagram

```
OpenStreetMap (Cottbus, Germany)
            │
            ▼
    OSMnx Graph Extraction
    (2128 nodes, ~5000 edges)
            │
            ▼
    Node Features: (lat, lon, degree)
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
 GraphSAGE    GraphSAGE
 (baseline)   (encoder for Transformer)
     │             │
     │         Node embeddings [N, 64]
     │             │
     │             ▼
     │       Trajectory window
     │       [B, seq_len=5, 64]
     │             │
     │             ▼
     │       Transformer Encoder
     │       (causal mask, 2 layers)
     │             │
     └──────┬──────┘
            │
            ▼
    Pointer scoring head:
    query · neighbour_embeddings
            │
            ▼
    Logits over local candidates
    (max_degree=8, padded)
            │
            ▼
    CrossEntropyLoss / argmax
```

---

## Project Structure

```
project/
├── src/
│   ├── data_loader.py          # OSMnx graph loading, random walks, train/val/test split
│   ├── dataset.py              # Neighbour-constrained dataset builder
│   ├── graph_model.py          # GraphSAGE with pointer scoring head
│   ├── transformer_model.py    # Transformer with pointer scoring head
│   ├── train_gnn.py            # GNN training script
│   └── train_transformers.py   # GNN + Transformer training script
│
├── notebooks/
│   ├── 01_load_graph.py
│   ├── 02_test_loader.py
│   ├── 03_generate_trajectories.py
│   ├── 04_build_dataset.py
│   ├── 05_test_gnn.py
│   └── 06_test_transformer.py
│
├── results/
│   ├── gnn_metrics.csv         # Per-epoch loss and accuracy (GNN)
│   └── transformer_metrics.csv # Per-epoch loss and accuracy (Transformer)
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Methodology

### 1. Graph Construction

Road network extracted from OpenStreetMap using OSMnx for Cottbus, Germany (`network_type='drive'`). Nodes represent intersections; edges represent driveable road segments. OSM node IDs are mapped to contiguous integer indices (0 to N-1) for embedding lookup.

Node features: `(latitude, longitude, degree)` — 3-dimensional.

### 2. Trajectory Generation

Random walks on the road graph with **anti-backtracking**: the previous node is excluded from the candidate pool at each step, producing more directionally consistent trajectories. 5,000 walks of length 8 are generated, then split 80/10/10 into train/val/test sets with a fixed random seed.

### 3. Dataset Construction

A sliding window of length `seq_len=5` is applied to each trajectory. For each window:
- The **input** is the sequence of 5 node indices
- The **candidates** are the neighbours of the last node in the window, padded to `max_degree=8` with `-1`
- The **target** is the local index (0–7) of the correct next node within the candidate list
- A **validity mask** marks real neighbours vs padding

### 4. Training

Both models are trained with:
- Adam optimiser, lr=0.001
- CrossEntropyLoss over local candidates
- Gradient clipping (max norm 1.0)
- 20 epochs

The Transformer additionally uses a causal upper-triangular attention mask so position `i` cannot attend to positions `i+1, ..., T-1`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Top-1 Accuracy | Exact next-node match |
| Top-3 Accuracy | Correct node in top-3 predictions |
| Random Baseline | `1 / avg_degree` — expected accuracy of uniform random choice |

The random baseline (~0.33–0.40 depending on local degree) is printed alongside every epoch so model improvement above chance is always visible.

---

## Expected Results

| Model | Top-1 | Top-3 | Notes |
|---|---|---|---|
| Random baseline | ~0.35 | ~0.80 | Uniform over neighbours |
| GNN (GraphSAGE) | ~0.40–0.45 | ~0.85 | Spatial only, no history |
| GNN + Transformer | ~0.50–0.65 | ~0.90 | Spatial + trajectory context |

The performance gap between GNN and Transformer quantifies the value of trajectory history for direction prediction at road intersections.

---

## Key Design Decisions

**Neighbour-constrained output space.** Predicting over all N nodes is unlearnable from synthetic random-walk data — the walk chooses uniformly among neighbours so there is no global pattern. Constraining to local candidates makes the task realistic and learnable.

**Shared pointer scoring head.** Both models use `query · candidate_embedding` scoring. This isolates the temporal dimension as the single architectural variable between models.

**End-to-end training.** The GNN encoder and Transformer are optimised jointly so spatial embeddings are shaped by the sequential prediction objective, not just neighbourhood aggregation.

**Causal attention mask.** The Transformer encoder is bidirectional by default. The causal mask prevents attention to future positions in the trajectory window, ensuring the model is genuinely predictive rather than interpolating.

---

## Limitations

- Trajectories are synthetic random walks, not real GPS traces
- No destination or intent signal in the trajectory
- Evaluation on a single mid-size German city (Cottbus)
- No temporal GNN baselines (e.g. T-GCN) included

---

## Future Work

- Replace synthetic walks with real GPS trajectory datasets (e.g. GeoLife, OpenStreetMap traces)
- Add Node2Vec biased random walks for more structured training data
- Integrate temporal GNN models (T-GCN, STGCN) as additional baselines
- Extend to multi-step prediction (predict full route, not just next node)
- Add map-matching evaluation: given noisy GPS points, recover the most likely road sequence

---

## Tech Stack

- Python 3.10+
- PyTorch
- PyTorch Geometric
- OSMnx
- NetworkX
- NumPy
- scikit-learn

---

## How to Run

### Install dependencies

```bash
pip install torch torch-geometric osmnx networkx numpy scikit-learn
```

### Install package

```bash
pip install -e .
```

### Train GNN baseline

```bash
python src/train_gnn.py
```

### Train GNN + Transformer

```bash
python src/train_transformers.py
```

Results are saved to `results/gnn_metrics.csv` and `results/transformer_metrics.csv`.

---

## Author

**Hamza Majid**
MSc Artificial Intelligence — Brandenburg University of Technology (BTU) Cottbus-Senftenberg