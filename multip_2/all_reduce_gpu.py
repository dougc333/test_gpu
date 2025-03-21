import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend="nccl")

# Model wrapped in DDP
model = nn.Linear(10, 10).cuda()
ddp_model = DDP(model)

# Forward pass
x = torch.randn(32, 10).cuda()
output = ddp_model(x)

# Backward pass (syncs gradients across processes)
output.mean().backward()

# DDP uses `all_reduce` internally to sync gradients