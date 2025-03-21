import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define the first part of the model (Conv layers on Worker 0 - GPU 0)
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(x.size(0), -1)  # Flatten

# Define the second part of the model (FC layers on Worker 1 - GPU 1)
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Define the Distributed Model using RPC
class DistributedModel(nn.Module):
    def __init__(self, worker_name):
        super().__init__()
        self.remote_fc = rpc.remote(worker_name, FCNet, args=())  # Remote model on Worker 1

    def forward(self, x):
        return self.remote_fc.rpc_sync().forward(x)

# Training function
def train(rank, world_size):
    """Main training loop"""
    # Initialize RPC
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    # Worker 0 (Rank 0) - Loads dataset & trains ConvNet
    if rank == 0:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=".", train=True, download=True, transform=transform),
            batch_size=64, shuffle=True
        )

        # Initialize model
        conv_model = ConvNet().cuda(0)  # Conv layers on GPU 0
        full_model = DistributedModel("worker1")  # Fully connected layers on Worker 1 (GPU 1)

        optimizer = optim.SGD(list(conv_model.parameters()), lr=0.01, momentum=0.9)

        for epoch in range(5):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(0), target.cuda(1)  # Data on GPU 0, labels on GPU 1
                
                optimizer.zero_grad()

                # Forward pass (ConvNet on GPU 0)
                conv_output = conv_model(data)

                # Forward pass (FCNet on GPU 1 via RPC)
                output = full_model(conv_output)

                # Compute loss (on GPU 1)
                loss = F.cross_entropy(output, target)

                # Backward pass
                loss.backward()

                # Sync gradients from FCNet to ConvNet
                rpc.rpc_sync("worker1", lambda: None)  # Ensure gradients are updated before optimizer step

                # Update ConvNet parameters
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    rpc.shutdown()  # Cleanup RPC

# Main execution
if __name__ == "__main__":
    world_size = 2  # Two processes (one per GPU)

    # Spawn two processes for distributed training
    mp.spawn(train, args=(world_size,), nprocs=world_size)