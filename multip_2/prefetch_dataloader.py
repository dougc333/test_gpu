# doesnt work on macos!


import multiprocessing as mp
import random
import time
import torch
from queue import Empty

# Settings
NUM_WORKERS = 2
BATCH_SIZE = 8
NUM_BATCHES = 10
QUEUE_SIZE = 4
DATA_SHAPE = (1, 28, 28)

def generate_fake_sample():
    """Simulate a single data sample like MNIST."""
    x = torch.randn(*DATA_SHAPE)
    y = torch.randint(0, 10, (1,))
    return x, y

def load_batch():
    """Simulate a full batch of samples."""
    xs, ys = [], []
    for _ in range(BATCH_SIZE):
        x, y = generate_fake_sample()
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.cat(ys)

def data_loader_worker(queue, done_event, rank, batches_per_worker):
    for i in range(batches_per_worker):
        # Simulate data prep time
        time.sleep(random.uniform(0.1, 0.4))
        x_batch, y_batch = load_batch()
        queue.put((x_batch, y_batch))  # Will block if queue is full
        print(f"[Worker {rank}] Put batch {i} (queue size: {queue.qsize()})")
    print(f"[Worker {rank}] Finished producing")
    done_event.set()  # Each worker sets its own done_event

def consumer(queue, done_events, num_batches_expected):
    num_received = 0
    while num_received < num_batches_expected:
        try:
            x_batch, y_batch = queue.get(timeout=1)
            print(f"[Trainer] Got batch {num_received} — x: {x_batch.shape}, y: {y_batch.shape}")
            time.sleep(random.uniform(0.2, 0.3))  # Simulate training step
            num_received += 1
        except Empty:
            if all(e.is_set() and queue.empty() for e in done_events):
                print("[Trainer] All done signals received and queue is empty. Exiting.")
                break
            continue
    print("[Trainer] Finished consuming all batches.")

def main():
    mp.set_start_method("spawn", force=True)

    queue = mp.Queue(maxsize=QUEUE_SIZE)  # Shared bounded queue = backpressure
    done_events = [mp.Event() for _ in range(NUM_WORKERS)]
    total_batches = NUM_BATCHES
    batches_per_worker = total_batches // NUM_WORKERS

    # Start loader workers
    workers = []
    for rank in range(NUM_WORKERS):
        p = mp.Process(target=data_loader_worker,
                       args=(queue, done_events[rank], rank, batches_per_worker))
        p.start()
        workers.append(p)

    # Start consumer
    consumer_proc = mp.Process(target=consumer,
                               args=(queue, done_events, total_batches))
    consumer_proc.start()

    # Wait for everything to finish
    for p in workers:
        p.join()
    consumer_proc.join()

    print("✅ Prefetching DataLoader complete.")

if __name__ == "__main__":
    main()