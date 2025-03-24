import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import os
import time

# Shared model class
class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Exposed method for worker to call
class ModelOwner:
    def __init__(self):
        self.model = SharedModel()
        self.lock = torch.multiprocessing.Lock()

    def train_step(self, step, rank):
        with self.lock:
            x = torch.randn(16, 10)
            y = torch.randint(0, 2, (16,))
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            criterion = nn.CrossEntropyLoss()

            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            return {
                "rank": rank,
                "step": step,
                "loss": loss.item(),
            }

# Worker calls into shared model
def worker_loop(shared_model_rref, rank, steps):
    for step in range(steps):
        result = rpc.rpc_sync(
            to=shared_model_rref.owner_name(),
            func=ModelOwner.train_step,
            args=(shared_model_rref, step, rank)
        )
        print(f"[Worker {rank}] Step {step} - Loss: {result['loss']:.4f}")
        time.sleep(1)

# Controller initializes model and RRefs
def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    rpc.init_rpc(name=f"worker{rank}" if rank > 0 else "controller", rank=rank, world_size=world_size)

    if rank == 0:
        owner = ModelOwner()
        shared_model_rref = rpc.RRef(owner)

        futures = []
        for i in range(1, world_size):
            fut = rpc.rpc_async(
                to=f"worker{i}",
                func=worker_loop,
                args=(shared_model_rref, i, 5)
            )
            futures.append(fut)
        for fut in futures:
            fut.wait()

    rpc.shutdown()

if __name__ == "__main__":
    world_size = 4  # 1 controller + 3 workers
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
