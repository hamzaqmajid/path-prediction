import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super(GraphSAGEModel, self).__init__()

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x, edge_index):
        # Step 1: message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Step 2: predict next node logits
        out = self.classifier(x)

        return out