import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, num_nodes, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(num_nodes, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(embed_dim, num_nodes)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)

        x = self.transformer(x)

        # take last token representation
        x = x[:, -1, :]

        out = self.fc(x)

        return out