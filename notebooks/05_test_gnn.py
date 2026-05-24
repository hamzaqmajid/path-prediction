import torch
from src.data_loader import load_graph, build_graph_data
from src.graph_model import GraphSAGEModel

# Step 1: load graph
G = load_graph("Cottbus, Germany")
x, edge_index, node_map = build_graph_data(G)

num_nodes = x.shape[0]
input_dim = x.shape[1]

# Step 2: model
model = GraphSAGEModel(input_dim, hidden_dim=64, num_nodes=num_nodes)

# Step 3: forward pass
out = model(x, edge_index)

print("Output shape:", out.shape)