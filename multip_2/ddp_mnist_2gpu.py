# using allreduce 2 processes on 2 gpus

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

# Define the model
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Distributed training function
def train(rank, world_size):
    """Main training function using DDP and all_reduce."""
    
    # Initialize Process Group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set device for this process

    # Define dataset & Distributed Sampler
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create Dataloader
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)

    # Create model, move to GPU, and wrap with DDP
    model = MNISTModel().cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Define loss and optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5):  # Train for 5 epochs
        sampler.set_epoch(epoch)  # Shuffle data differently in each epoch

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank, non_blocking=True), target.cuda(rank, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Perform all_reduce manually (normally handled inside DDP)
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size  # Average gradients manually

            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    dist.destroy_process_group()  # Cleanup process group

# Main function
if __name__ == "__main__":
    world_size = 2  # Number of GPUs

    # Set environment variables for distributed execution
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Spawn multiple processes (one per GPU)
    mp.spawn(train, args=(world_size,), nprocs=world_size)