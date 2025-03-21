import torch
import torch.multiprocessing as mp
import time
import random

# Constants
BATCH_SIZE = 64
IMG_SHAPE = (1, 28, 28)
NUM_WORKERS = 2
TOTAL_BATCHES = 10

def create_shared_batch():
    """Create a shared-memory tensor batch (x, y)."""
    x = torch.empty(BATCH_SIZE, *IMG_SHAPE).share_memory_()
    y = torch.empty(BATCH_SIZE, dtype=torch.long).share_memory_()
    return x, y

def fill_batch(x, y):
    """Simulate filling a batch like MNIST."""
    x.copy_(torch.randn_like(x))
    y.copy_(torch.randint(0, 10, y.shape))

def data_loader_worker(rank, shared_queue, num_batches):
    for i in range(num_batches):
        x, y = create_shared_batch()
        fill_batch(x, y)
        shared_queue.put((x, y))
        print(f"[Loader {rank}] Put batch {i}")
    shared_queue.put(None)  # Sentinel for completion

def trainer(num_workers, total_batches):
    manager = mp.Manager()
    shared_queue = manager.Queue(maxsize=8)

    batches_per_worker = total_batches // num_workers
    workers = []

    for rank in range(num_workers):
        p = mp.Process(target=data_loader_worker, args=(rank, shared_queue, batches_per_worker))
        p.start()
        workers.append(p)

    # Main loop to consume batches
    done_count = 0
    while done_count < num_workers:
        item = shared_queue.get()
        if item is None:
            done_count += 1
            continue
        x, y = item
        print(f"[Trainer] Got batch with shape: {x.shape}, label shape: {y.shape}, mean: {x.mean():.4f}")

    for p in workers:
        p.join()

    print("✅ Training done with Manager-based shared queue")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    trainer(num_workers=NUM_WORKERS, total_batches=TOTAL_BATCHES)