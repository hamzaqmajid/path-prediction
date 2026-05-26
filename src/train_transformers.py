import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.transformer_model import TransformerModel
from src.data_loader import build_graph_data, load_graph, generate_trajectories


# -----------------------
# PREPARE DATA
# -----------------------

def prepare_data(trajectories, seq_len=5):
    X = []
    y = []

    for traj in trajectories:
        for i in range(len(traj) - seq_len):
            X.append(traj[i:i+seq_len])
            y.append(traj[i+seq_len])

    return torch.tensor(X), torch.tensor(y)


# -----------------------
# TRAIN
# -----------------------

def train():
    num_nodes = 2128
    G = load_graph("Cottbus, Germany")
    x, edge_index, node_map = build_graph_data(G)
    trajectories = generate_trajectories(G, node_map)
    X, y = prepare_data(trajectories)

    device = torch.device("cpu")

    model = TransformerModel(num_nodes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X = X.to(device)
    y = y.to(device)

    epochs = 20

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()