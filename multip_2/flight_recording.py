import torch
import torch.distributed as dist

# Initialize Distributed Process Group
dist.init_process_group(backend="nccl")

# Enable FlightRecorder
torch.distributed.FlightRecorder.enable()

# Run some collective operations
tensor = torch.randn(10).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Save logs for later analysis
torch.distributed.FlightRecorder.dump("flightrecorder_logs.json")

# Disable after profiling
torch.distributed.FlightRecorder.disable()

