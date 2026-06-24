"""
graph_model.py — GraphSAGE with neighbour-constrained classification head.

Architecture change
───────────────────
Original: Linear(hidden_dim → num_nodes)  — predict over all 2 128 nodes.
New:      score each candidate neighbour by dot-product with the query
          node embedding, then softmax over valid candidates only.

This is called a "pointer network" style head: instead of a fixed output
vocabulary, the model scores a dynamic set of candidates.  It naturally
handles variable-degree nodes and makes the comparison with the Transformer
fair (both use the same scoring mechanism).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, max_degree=8):
        """
        Parameters
        ----------
        input_dim  : node feature dimension (lat, lon, degree → 3)
        hidden_dim : GNN hidden / output dimension
        num_nodes  : total nodes in graph (kept for compatibility)
        max_degree : padded candidate list length (must match dataset.py)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_degree = max_degree

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # Projection applied to the query (current node) embedding before
        # scoring against candidate embeddings.  Adds learnable capacity
        # beyond raw dot-product.
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward_embeddings(self, x, edge_index):
        """
        Compute spatial node embeddings via two GraphSAGE layers.

        Returns
        -------
        Tensor [num_nodes, hidden_dim]
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index, input_seqs, candidates, valid_masks):
        """
        Neighbour-constrained next-node prediction.

        Parameters
        ----------
        x           : FloatTensor [N, input_dim]       — node features
        edge_index  : LongTensor  [2, E]               — graph edges
        input_seqs  : LongTensor  [B, seq_len]         — trajectory windows
        candidates  : LongTensor  [B, max_degree]      — neighbour indices (-1=pad)
        valid_masks : BoolTensor  [B, max_degree]      — True = real neighbour

        Returns
        -------
        logits : FloatTensor [B, max_degree]
                 Masked so padding positions are -inf before softmax.
        """
        node_emb = self.forward_embeddings(x, edge_index)  # [N, D]

        # Query = embedding of the last node in each trajectory window
        query_nodes = input_seqs[:, -1]                    # [B]
        query_emb   = self.query_proj(node_emb[query_nodes])  # [B, D]

        # Candidate embeddings — clamp -1 padding to 0 to avoid index error
        # (padding positions are masked out before loss/argmax anyway)
        safe_candidates = candidates.clamp(min=0)          # [B, max_degree]
        cand_emb = node_emb[safe_candidates]               # [B, max_degree, D]

        # Dot-product score: query · each candidate
        # query_emb unsqueezed → [B, 1, D], bmm → [B, 1, max_degree], squeeze
        scores = torch.bmm(query_emb.unsqueeze(1), cand_emb.transpose(1, 2))
        scores = scores.squeeze(1)                         # [B, max_degree]

        # Mask padding positions to -inf so they never win argmax / softmax
        scores = scores.masked_fill(~valid_masks, float("-inf"))

        return scores