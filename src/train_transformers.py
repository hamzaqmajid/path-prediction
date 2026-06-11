import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.transformer_model import TransformerModel
from src.data_loader import build_graph_data, load_graph, generate_trajectories
from sklearn.metrics import confusion_matrix

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

def compute_accuracy(output, y):
    preds = torch.argmax(output, dim=1)
    correct = (preds == y).sum().item()
    total = y.size(0)
    return correct / total

def compute_confusion_matrix(output, y, num_classes=50):
    preds = torch.argmax(output, dim=1).cpu().numpy()
    y_true = y.cpu().numpy()

    # take subset of classes to keep it small
    mask = y_true < num_classes

    cm = confusion_matrix(y_true[mask], preds[mask])

    return cm

def train():
    num_nodes = 2128
    G = load_graph("Cottbus, Germany")
    x, edge_index, node_map = build_graph_data(G)
    trajectories = generate_trajectories(G, node_map)
    X, y = prepare_data(trajectories)

    device = torch.device("cpu")
    print(object.__str__(X.shape))

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
        acc = compute_accuracy(output, y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")
    cm = compute_confusion_matrix(output, y)
    print("\nConfusion Matrix (partial):")
    print(cm)




if __name__ == "__main__":
    train()