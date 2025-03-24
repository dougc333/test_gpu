import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import os
import time
from collections import defaultdict

CHECKPOINT_PATH = "controller_model.pt"
SYNC_INTERVAL = 3

# Shared model
class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

    def forward(self, x):
        return self.net(x)

# Model holder for RRef access
class ModelOwner:
    def __init__(self):
        self.model = SharedModel()
        self.lock = torch.multiprocessing.Lock()
        self.loss_log = []
        if os.path.exists(CHECKPOINT_PATH):
            self.model.load_state_dict(torch.load(CHECKPOINT_PATH))
            print("[ModelOwner] Loaded checkpoint.")

    def train_step(self, rank, step):
        with self.lock:
            x = torch.randn(16, 10)
            y = torch.randint(0, 2, (16,))
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            loss_fn = nn.CrossEntropyLoss()
            out = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            self.loss_log.append((step, loss.item()))
            torch.save({
                'model': self.model.state_dict(),
                'step': step,
                'loss': loss.item(),
                'timestamp': time.time()
            }, CHECKPOINT_PATH)

            return {"rank": rank, "step": step, "loss": loss.item()}

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_loss_log(self):
        return self.loss_log

# Average parameters between two models
def average_state_dicts(state1, state2):
    avg_state = {}
    for key in state1:
        avg_state[key] = (state1[key] + state2[key]) / 2
    return avg_state

# Worker logic
def worker_loop(rank, controller_names, steps):
    for step in range(steps):
        controller = controller_names[step % len(controller_names)]
        result = rpc.rpc_sync(
            to=controller,
            func=ModelOwner.train_step,
            args=(rank, step)
        )
        print(f"[Worker {rank}] Step {step} via {controller} → Loss: {result['loss']:.4f}")
        time.sleep(1)

# Synchronize and average controller models
def sync_controllers(my_name, peer_name):
    my_rref = rpc.RRef(ModelOwner())
    peer_state = rpc.rpc_sync(peer_name, ModelOwner.get_state_dict, args=())
    my_state = my_rref.local_value().get_state_dict()
    avg_state = average_state_dicts(my_state, peer_state)
    my_rref.local_value().load_state_dict(avg_state)
    torch.save(avg_state, CHECKPOINT_PATH)
    print(f"[{my_name}] Averaged and saved model from {peer_name}")

# Compare loss between synced/unsynced models
def compare_loss_logs():
    loss_c0 = rpc.rpc_sync("controller0", ModelOwner.get_loss_log, args=())
    loss_c1 = rpc.rpc_sync("controller1", ModelOwner.get_loss_log, args=())
    print("\n📊 Loss log comparison:")
    for l0, l1 in zip(loss_c0, loss_c1):
        print(f"Step {l0[0]} - Ctrl0: {l0[1]:.4f}, Ctrl1: {l1[1]:.4f}")

# Each controller owns a model instance
def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29600"

    name = f"controller{rank}" if rank < 2 else f"worker{rank}"
    rpc.init_rpc(name=name, rank=rank, world_size=world_size)

    if name == "controller0":
        owner = ModelOwner()
        time.sleep(5)  # wait for controller1 to initialize

        for sync_step in range(0, 5, SYNC_INTERVAL):
            time.sleep(SYNC_INTERVAL * 1.1)
            sync_controllers("controller0", "controller1")

        compare_loss_logs()

    elif name == "controller1":
        ModelOwner()

    elif name.startswith("worker"):
        controller_names = [f"controller{i}" for i in range(2)]
        worker_loop(rank, controller_names, steps=5)

    rpc.shutdown()

if __name__ == "__main__":
    world_size = 5  # 2 controllers + 3 workers
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)