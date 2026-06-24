"""
transformer_model.py — Transformer with neighbour-constrained scoring head.

The Transformer reads a trajectory window and produces a context vector.
That vector is then scored against each candidate neighbour's GNN embedding
via dot-product — the same pointer-style head used in GraphSAGEModel —
so the two models are directly comparable.
"""

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        embed_dim=64,
        max_len=100,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_degree=8,
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.max_degree = max_degree

        # Project GNN embeddings into transformer space
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project the last-token context vector before scoring candidates
        self.query_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, node_emb, candidates, valid_masks, mask=None):
        """
        Parameters
        ----------
        x           : FloatTensor [B, seq_len, embed_dim]  — embedded trajectory
        node_emb    : FloatTensor [N, embed_dim]           — all node embeddings
        candidates  : LongTensor  [B, max_degree]          — neighbour indices
        valid_masks : BoolTensor  [B, max_degree]          — True = real neighbour
        mask        : FloatTensor [seq_len, seq_len]       — causal attention mask

        Returns
        -------
        logits : FloatTensor [B, max_degree]   — masked, ready for CrossEntropy
        """
        _, seq_len, _ = x.shape

        # Project + add positional embeddings
        x = self.input_proj(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)              # [B, T, D]

        # Transformer encoder with causal mask
        x = self.transformer(x, mask=mask)                 # [B, T, D]

        # Use last token as the trajectory context vector
        context = self.query_proj(x[:, -1, :])             # [B, D]

        # Score each candidate neighbour via dot-product with context
        safe_candidates = candidates.clamp(min=0)          # [B, max_degree]
        cand_emb = node_emb[safe_candidates]               # [B, max_degree, D]

        scores = torch.bmm(context.unsqueeze(1), cand_emb.transpose(1, 2))
        scores = scores.squeeze(1)                         # [B, max_degree]

        # Mask padding positions to -inf so they never influence loss/argmax
        scores = scores.masked_fill(~valid_masks, float("-inf"))

        return scores