import torch

# Create a tensor
x = torch.tensor([[10, 20], [30, 40]])

# Gather indices
indices = torch.tensor([[0, 1], [1, 0]])

# Gather values from x based on indices
gathered = torch.gather(x, dim=1, index=indices)

print(gathered)