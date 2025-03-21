import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

# Simple CNN for MNIST
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

# Manual gradient averaging across processes
def average_gradients(model, world_size):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size  # Average gradients

def train(rank, world_size):
    print(f"Starting training on rank {rank}")
    
    # Set up environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Prepare dataset and dataloader with DistributedSampler
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Create model, move to correct GPU
    model = CNN().cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    model.train()
    for epoch in range(5):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()

            # Manually sync gradients
            average_gradients(model, world_size)

            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"[Rank {rank}] Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Number of processes / GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size)