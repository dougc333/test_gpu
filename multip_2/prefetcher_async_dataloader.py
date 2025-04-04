import torch
import torch.multiprocessing as mp
import time
import random
from queue import Empty

class AsyncPrefetchLoader:
    def __init__(
        self,
        num_workers=2,
        batch_size=64,
        num_batches=100,
        queue_size=8,
        data_shape=(1, 28, 28),
        device="cpu",
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.queue_size = queue_size
        self.data_shape = data_shape
        self.device = device

        self.queue = mp.Queue(maxsize=queue_size)
        self.done_events = [mp.Event() for _ in range(num_workers)]
        self.total_batches = num_batches
        self.batches_per_worker = num_batches // num_workers
        self.workers = []
        self._start_workers()

    def _generate_fake_batch(self):
        x = torch.randn(self.batch_size, *self.data_shape)
        y = torch.randint(0, 10, (self.batch_size,))
        return x.to(self.device), y.to(self.device)

    def _worker_fn(self, queue, done_event, rank, batches_per_worker):
        for i in range(batches_per_worker):
            time.sleep(random.uniform(0.1, 0.3))  # Simulate I/O
            x, y = self._generate_fake_batch()
            queue.put((x, y))  # blocks if queue is full
            print(f"[Worker {rank}] Put batch {i}")
        done_event.set()
        print(f"[Worker {rank}] Done")

    def _start_workers(self):
        for rank in range(self.num_workers):
            p = mp.Process(
                target=self._worker_fn,
                args=(self.queue, self.done_events[rank], rank, self.batches_per_worker),
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def __iter__(self):
        self.batches_returned = 0
        return self

    def __next__(self):
        if self.batches_returned >= self.total_batches:
            raise StopIteration

        while True:
            try:
                batch = self.queue.get(timeout=1)
                self.batches_returned += 1
                return batch
            except Empty:
                if all(e.is_set() and self.queue.empty() for e in self.done_events):
                    raise StopIteration

    def shutdown(self):
        for p in self.workers:
            p.join()

# ---------------------
# 🧪 Example Usage
# ---------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    loader = AsyncPrefetchLoader(
        num_workers=2,
        batch_size=32,
        num_batches=10,
        queue_size=4,
        data_shape=(1, 28, 28),
        device="cpu",
    )

    for i, (x, y) in enumerate(loader):
        print(f"[Main] Got batch {i} — x: {x.shape}, y: {y.shape}")
        time.sleep(0.2)  # Simulate training step

    loader.shutdown()
    print("✅ Done with AsyncPrefetchLoader")