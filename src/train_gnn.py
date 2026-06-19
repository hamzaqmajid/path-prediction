import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import load_graph, build_graph_data, generate_trajectories
from src.dataset import build_sequence_dataset
from src.graph_model import GraphSAGEModel

from sklearn.metrics import confusion_matrix

# -----------------------
# Load graph
# -----------------------
G = load_graph("Cottbus, Germany")
x, edge_index, node_map = build_graph_data(G)

num_nodes = x.shape[0]
input_dim = x.shape[1]

# -----------------------
# Generate data
# -----------------------
trajectories = generate_trajectories(G, node_map)
inputs, targets = build_sequence_dataset(trajectories)

# -----------------------
# Model
# -----------------------
device = torch.device("cpu")
model = GraphSAGEModel(input_dim, hidden_dim=64, num_nodes=num_nodes).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def compute_accuracy(output, y):
    preds = torch.argmax(output, dim=1)
    correct = (preds == y).sum().item()
    total = y.size(0)
    return correct / total

def compute_confusion_matrix_from_lists(preds, targets):
    return confusion_matrix(targets, preds)

def compute_confusion_matrix(output, y, num_classes=50):
    preds = torch.argmax(output, dim=1).cpu().numpy()
    y_true = y.cpu().numpy()

    mask = y_true < num_classes

    if mask.sum() == 0:
        print("Warning: No samples match the mask. Returning empty matrix.")
        return None

    cm = confusion_matrix(y_true[mask], preds[mask])
    return cm
# -----------------------
# Training loop
# -----------------------

all_preds = []
all_targets = []


for epoch in range(10):
    total_loss = 0

    for inp, target in zip(inputs, targets):

        inp_node = inp[-1]  # last visited node = current state

        x_device = x.to(device)
        edge_device = edge_index.to(device)

        out = model(x_device, edge_device)

        pred = out[inp_node].unsqueeze(0)  # prediction for current node
        target_tensor = torch.tensor([target], device=device)

        loss = loss_fn(pred, target_tensor)
        acc = compute_accuracy(pred, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(torch.argmax(pred).item())
        all_targets.append(target)
        epoch_acc = sum(
            1 for p, t in zip(all_preds, all_targets) if p == t
        ) / len(all_targets)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {epoch_acc:.4f}")
cm = compute_confusion_matrix_from_lists(all_preds, all_targets)
print("\nConfusion Matrix (partial):")
print(cm)