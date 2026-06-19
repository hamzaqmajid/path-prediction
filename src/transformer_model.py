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
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.max_len = max_len

        # projection (safe even if already same dim)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # -----------------------
        # POSITIONAL EMBEDDINGS (LEARNED)
        # -----------------------
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(embed_dim, num_nodes)

    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """

        batch_size, seq_len, _ = x.shape

        # projection
        x = self.input_proj(x)

        # -----------------------
        # ADD POSITIONAL EMBEDDINGS
        # -----------------------
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]

        pos_emb = self.pos_embedding(positions)  # [1, seq_len, embed_dim]

        x = x + pos_emb  # broadcast to batch

        # transformer
        x = self.transformer(x)

        # use last token
        out = x[:, -1, :]

        logits = self.fc_out(out)

        return logits