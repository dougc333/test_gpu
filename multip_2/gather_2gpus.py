import torch

# Simulating tensors from 2 GPUs
tensor_1 = torch.tensor([10, 30], device="cuda:0")
tensor_2 = torch.tensor([20, 40], device="cuda:1")

# Move tensors to CPU for gathering
tensor_1 = tensor_1.cpu()
tensor_2 = tensor_2.cpu()

# Gather the tensors
gathered_tensor = torch.cat([tensor_1, tensor_2])

print("Gathered Tensor:", gathered_tensor)