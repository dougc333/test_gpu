import torch

import torch.distributed as dist

# List of common backends
backends = ["gloo", "nccl", "mpi"]

available_backends = {backend: dist.is_nccl_available() if backend == "nccl" else dist.is_gloo_available() if backend == "gloo" else dist.is_mpi_available() for backend in backends}

print("Available Distributed Backends:")
for backend, available in available_backends.items():
    print(f"{backend}: {'✅ Available' if available else '❌ Not Available'}")