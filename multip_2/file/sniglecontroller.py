#https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=39
#15.20

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os

# Simple model
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

    def forward(self, x):
        return self.net(x)

# Worker function: receives step, runs training
def train_step(model_rref, step, rank):
    model = model_rref.local_value()
    model.train()

    x = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    start = time.time()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    sync_time = time.time() - start

    return {
        "rank": rank,
        "step": step,
        "loss": loss.item(),
        "sync_time": sync_time
    }

# Controller logic
def controller_loop(worker_names, steps=5):
    model_rrefs = {name: rpc.remote(name, ToyModel) for name in worker_names}

    for step in range(steps):
        futures = []
        print(f"\n🧠 Controller: Step {step}")
        for name in worker_names:
            fut = rpc.rpc_async(
                to=name,
                func=train_step,
                args=(model_rrefs[name], step, int(name[-1]))
            )
            futures.append(fut)

        results = [f.wait() for f in futures]
        for r in results:
            print(f"[Worker {r['rank']}] Loss: {r['loss']:.4f}, Sync: {r['sync_time']:.3f}s")

# Entry point
def run(rank, world_size):
    name = f"worker{rank}" if rank > 0 else "controller"
    rpc.init_rpc(name=name, rank=rank, world_size=world_size)

    if rank == 0:
        worker_names = [f"worker{i}" for i in range(1, world_size)]
        controller_loop(worker_names)

    rpc.shutdown()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    from sys import argv

    world_size = 4  # 1 controller + 3 workers
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)