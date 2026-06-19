import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super().__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # classifier head (ONLY for GNN baseline)
        self.classifier = nn.Linear(hidden_dim, num_nodes)

    # MODE 1: embeddings (for transformer)
    def forward_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x  # (N, 64)

    #  MODE 2: classification (for standalone GNN)
    def forward(self, x, edge_index):
        x = self.forward_embeddings(x, edge_index)
        x = self.classifier(x)
        return x  # (N, num_nodes)