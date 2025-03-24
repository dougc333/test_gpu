#Avoid mp.set_start_method() globally if you’re building reusable or testable libraries. Use get_context() + multiprocessing_context= instead.
# 
# #

import torch
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import os
import time

# ✅ Simple dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate work
        time.sleep(0.01)
        return torch.tensor(idx)

# ✅ Function to build dataloader with selected start method
def create_dataloader(start_method="fork", num_workers=2):
    print(f"\n🚀 Using start method: {start_method}")
    
    # Set context (instead of global set_start_method)
    ctx = mp.get_context(start_method)
    
    dataset = DummyDataset()
    
    return DataLoader(
        dataset,
        batch_size=10,
        num_workers=num_workers,
        multiprocessing_context=ctx,  # 👈 This enables dynamic control
        shuffle=False,
    )

# ✅ Main training loop
def run_loader(dataloader):
    for i, batch in enumerate(dataloader):
        print(f"[Batch {i}] Data: {batch.tolist()}")

if __name__ == "__main__":
    # Choose start method: "fork", "spawn", or "forkserver"
    start_method = "spawn" if torch.cuda.is_available() else "fork"

    dataloader = create_dataloader(start_method=start_method, num_workers=2)
    run_loader(dataloader)