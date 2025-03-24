# Run this with torchrun --nproc_per_node=2 cpu_allreduce.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    tensor = torch.ones(3) * (rank + 1)  # Rank 0: [1,1,1], Rank 1: [2,2,2]
    print(f"[Rank {rank}] Before all_reduce: {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[Rank {rank}] After all_reduce: {tensor}")  # Both will see [3,3,3]

    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2)