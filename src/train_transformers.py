import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.graph_model import GraphSAGEModel
from src.transformer_model import TransformerModel
from src.data_loader import build_graph_data, load_graph, generate_trajectories
from sklearn.metrics import confusion_matrix


# -----------------------
# DATA PREP
# -----------------------

def prepare_data(trajectories, seq_len=5):
    X, y = [], []

    for traj in trajectories:
        for i in range(len(traj) - seq_len):
            X.append(traj[i:i+seq_len])
            y.append(traj[i+seq_len])

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# -----------------------
# METRICS
# -----------------------

def compute_accuracy(output, y):
    preds = torch.argmax(output, dim=1)
    return (preds == y).float().mean().item()


def compute_confusion_matrix(output, y, num_classes=50):
    preds = torch.argmax(output, dim=1).cpu().numpy()
    y_true = y.cpu().numpy()

    mask = y_true < num_classes

    if mask.sum() == 0:
        return np.zeros((num_classes, num_classes))

    return confusion_matrix(
        y_true[mask],
        preds[mask],
        labels=list(range(num_classes))
    )


# -----------------------
# TRAIN LOOP
# -----------------------

def train():
    device = torch.device("cpu")

    # -----------------------
    # Load graph
    # -----------------------
    G = load_graph("Cottbus, Germany")
    x, edge_index, node_map = build_graph_data(G)

    num_nodes = x.size(0)

    # -----------------------
    # Trajectories → dataset
    # -----------------------
    trajectories = generate_trajectories(G, node_map)
    X, y = prepare_data(trajectories)

    X = X.to(device)
    y = y.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    # -----------------------
    # Models
    # -----------------------
    gnn = GraphSAGEModel(
        input_dim=x.size(1),
        hidden_dim=64,
        num_nodes=num_nodes
    ).to(device)

    model = TransformerModel(
        num_nodes=num_nodes,
        embed_dim=64
    ).to(device)

    # IMPORTANT: optimizer includes both models
    optimizer = optim.Adam(
        list(gnn.parameters()) + list(model.parameters()),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    epochs = 20

    # -----------------------
    # TRAINING
    # -----------------------
    for epoch in range(epochs):

        gnn.train()
        model.train()

        # --- Graph embeddings (kept differentiable) ---
        node_embeddings = gnn.forward_embeddings(x, edge_index)  # [N, D]

        # --- Build sequences ---
        embedded_sequences = node_embeddings[X]  # [B, T, D]

        # --- Forward ---
        output = model(embedded_sequences)  # [B, num_nodes]

        loss = criterion(output, y)
        acc = compute_accuracy(output, y)

        # --- Backprop ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    # -----------------------
    # FINAL EVALUATION
    # -----------------------
    gnn.eval()
    model.eval()

    with torch.no_grad():
        node_embeddings = gnn.forward_embeddings(x, edge_index)
        embedded_sequences = node_embeddings[X]
        output = model(embedded_sequences)

    cm = compute_confusion_matrix(output, y)

    print("\nConfusion Matrix (partial):")
    print(cm)


if __name__ == "__main__":
    train()