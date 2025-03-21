import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch import nn

# Define a model partitioned across multiple nodes
class ModelShard(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)

# Define the full model that calls RPC
class DistributedModel(nn.Module):
    def __init__(self, worker_name):
        super().__init__()
        self.remote_model = rpc.remote(worker_name, ModelShard, args=(10, 5))

    def forward(self, x):
        return self.remote_model.rpc_sync().forward(x)

# Worker function
def run_worker(rank, world_size):
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    if rank == 0:
        model = DistributedModel("worker1")  # Use remote model on worker1
        x = torch.randn(4, 10)  # Batch size of 4
        output = model(x)
        print("Output:", output)

    rpc.shutdown()

if __name__ == "__main__":
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = 2  # Two processes: one for main model, one for remote shard
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size)