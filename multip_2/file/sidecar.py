import os
import time
import json
import socket
import torch

# Simulated global registry (could be replaced by real training hook)
ddp_stats = {}

def update_stats(rank, loss, sync_time, step, device):
    hostname = socket.gethostname()
    ddp_stats[(hostname, rank)] = {
        "rank": rank,
        "device": device,
        "step": step,
        "loss": loss,
        "sync_time": sync_time,
        "hostname": hostname,
    }

def write_stats_loop(filepath="./tmp/ddp_stats.json", interval=2.0):
    while True:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(list(ddp_stats.values()), f, indent=2)
        time.sleep(interval)

# Example of injecting stats
if __name__ == "__main__":
    import threading
    threading.Thread(target=write_stats_loop, daemon=True).start()

    # Simulated loop for 2 ranks
    for step in range(50):
        for rank in range(2):
            update_stats(
                rank=rank,
                loss=torch.randn(1).abs().item(),
                sync_time=0.05 + 0.03 * torch.rand(1).item(),
                step=step,
                device=f"cuda:{rank}"
            )
        time.sleep(2)