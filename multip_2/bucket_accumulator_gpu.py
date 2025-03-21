import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group("nccl")

# Define a simple model
model = nn.Linear(1024, 1024).cuda()

# Wrap with DDP (automatic bucketing happens inside)
ddp_model = DDP(model, bucket_cap_mb=25)  # Adjusting bucket size

# Forward pass
x = torch.randn(32, 1024).cuda()
output = ddp_model(x)

# Backward pass
output.mean().backward()