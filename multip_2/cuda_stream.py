import torch
import time

# Create two CUDA streams
compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

# Dummy CPU data
cpu_data1 = torch.randn(10000, 1000, dtype=torch.float32, pin_memory=True)  # Use pinned memory
cpu_data2 = torch.randn(10000, 1000, dtype=torch.float32, pin_memory=True)

# Move model to GPU
device = torch.device("cuda")
model = torch.nn.Linear(1000, 1000).to(device)

# Start timing
start_time = time.time()

# Begin first data transfer (async)
with torch.cuda.stream(transfer_stream):
    gpu_data1 = cpu_data1.to(device, non_blocking=True)  # Async transfer

# Start computation on GPU while transferring the second batch
with torch.cuda.stream(compute_stream):
    output1 = model(gpu_data1)  # Compute on first batch

# Now transfer the second batch while first batch is computing
with torch.cuda.stream(transfer_stream):
    gpu_data2 = cpu_data2.to(device, non_blocking=True)

# Synchronize streams before using the second batch
torch.cuda.synchronize()

# Compute on second batch
output2 = model(gpu_data2)

# End timing
end_time = time.time()

print(f"Total time with overlapping: {end_time - start_time:.4f} seconds")