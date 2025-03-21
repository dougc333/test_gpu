import torch
import torch.multiprocessing as mp
import time
import random

# Constants
BATCH_SIZE = 64
NUM_WORKERS = 2
TOTAL_BATCHES = 10
IMG_SHAPE = (1, 28, 28)

def create_shared_batch():
    """Create a shared-memory tensor batch (x, y)."""
    x = torch.empty(BATCH_SIZE, *IMG_SHAPE).share_memory_()
    y = torch.empty(BATCH_SIZE, dtype=torch.long).share_memory_()
    return x, y

def fill_batch(x, y):
    """Simulate loading MNIST-style data into tensors."""
    x.copy_(torch.randn_like(x))
    y.copy_(torch.randint(0, 10, y.shape))

def data_loader_worker(rank, queue, num_batches):
    for i in range(num_batches):
        x, y = create_shared_batch()
        fill_batch(x, y)
        queue.put((x, y))
        print(f"[Loader {rank}] Sent shared batch {i}")
    queue.put(None)  # Signal done

def trainer(num_workers, total_batches):
    queue = mp.Queue(maxsize=4)
    batches_per_worker = total_batches // num_workers

    # Start worker processes
    workers = []
    for rank in range(num_workers):
        p = mp.Process(target=data_loader_worker, args=(rank, queue, batches_per_worker))
        p.start()
        workers.append(p)

    # Training loop (consume shared-memory batches)
    done_count = 0
    while done_count < num_workers:
        item = queue.get()
        if item is None:
            done_count += 1
            continue
        x, y = item
        # No copy — x and y are already in shared memory!
        print(f"[Trainer] Got batch: x={x.shape}, y={y.shape}, mean={x.mean():.4f}")

    for p in workers:
        p.join()

    print("✅ Training complete with shared memory!")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Use 'spawn' for cross-platform compatibility
    trainer(num_workers=NUM_WORKERS, total_batches=TOTAL_BATCHES)