import torch
from src.transformer_model import TransformerModel

num_nodes = 2128

model = TransformerModel(num_nodes)

# fake batch: 2 sequences of length 5
x = torch.randint(0, num_nodes, (2, 5))

out = model(x)

print("Output shape:", out.shape)