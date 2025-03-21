import torch
import torch.multiprocessing as mp
import time
import random

# Config
BUFFER_SIZE = 4  # Number of preloaded batches
BATCH_SIZE = 64
IMG_SHAPE = (1, 28, 28)
NUM_BATCHES = 8

def create_ring_buffer(size):
    """Creates shared-memory buffers for x and y."""
    x_buffer = [torch.empty(BATCH_SIZE, *IMG_SHAPE).share_memory_() for _ in range(size)]
    y_buffer = [torch.empty(BATCH_SIZE, dtype=torch.long).share_memory_() for _ in range(size)]
    return x_buffer, y_buffer

def producer(x_buf, y_buf, head, tail, count, lock, empty_sem, full_sem):
    for i in range(NUM_BATCHES):
        empty_sem.acquire()  # Wait for space
        with lock:
            idx = head.value % BUFFER_SIZE
            x_buf[idx].copy_(torch.randn_like(x_buf[idx]))
            y_buf[idx].copy_(torch.randint(0, 10, y_buf[idx].shape))
            print(f"[Loader] Wrote batch {i} to slot {idx}")
            head.value += 1
            count.value += 1
        full_sem.release()  # Signal data is available
    print("[Loader] Done")

def consumer(x_buf, y_buf, head, tail, count, lock, empty_sem, full_sem):
    for _ in range(NUM_BATCHES):
        full_sem.acquire()  # Wait for data
        with lock:
            idx = tail.value % BUFFER_SIZE
            x = x_buf[idx]
            y = y_buf[idx]
            print(f"[Trainer] Got batch from slot {idx}, mean={x.mean():.4f}")
            tail.value += 1
            count.value -= 1
        empty_sem.release()  # Signal space is available
    print("[Trainer] Done")

def main():
    mp.set_start_method("spawn", force=True)

    # Create ring buffer
    x_buf, y_buf = create_ring_buffer(BUFFER_SIZE)

    # Shared counters and locks
    head = mp.Value('i', 0)  # Write index
    tail = mp.Value('i', 0)  # Read index
    count = mp.Value('i', 0) # Current buffer size
    lock = mp.Lock()
    empty_sem = mp.Semaphore(BUFFER_SIZE)  # Empty slots
    full_sem = mp.Semaphore(0)             # Filled slots

    # Launch producer and consumer
    p_loader = mp.Process(target=producer, args=(x_buf, y_buf, head, tail, count, lock, empty_sem, full_sem))
    p_trainer = mp.Process(target=consumer, args=(x_buf, y_buf, head, tail, count, lock, empty_sem, full_sem))

    p_loader.start()
    p_trainer.start()
    p_loader.join()
    p_trainer.join()

    print("✅ Done with ring buffer data loading!")

if __name__ == "__main__":
    main()