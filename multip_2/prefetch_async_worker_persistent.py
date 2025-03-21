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
        batches_per_epoch=10,
        queue_size=8,
        data_shape=(1, 28, 28),
        device="cpu",
        max_retries=3,
        persistent_workers=True,
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.data_shape = data_shape
        self.device = device
        self.max_retries = max_retries
        self.persistent_workers = persistent_workers
        self.batches_per_epoch = batches_per_epoch

        self.queue = mp.Queue(maxsize=queue_size)
        self.shutdown_event = mp.Event()  # Global shutdown
        self.control_pipes = []  # Main <-> Worker control pipes
        self.workers = []
        self._start_workers()

    def _generate_fake_batch(self):
        if random.random() < 0.1:
            raise RuntimeError("Simulated data loading error")
        x = torch.randn(self.batch_size, *self.data_shape)
        y = torch.randint(0, 10, (self.batch_size,))
        return x.to(self.device), y.to(self.device)

    def _worker_fn(self, queue, pipe, shutdown_event):
        """Worker that waits for an epoch command, then loads N batches."""
        while not shutdown_event.is_set():
            try:
                command, num_batches = pipe.recv()  # Wait for epoch command
                if command == "run":
                    for i in range(num_batches):
                        retries = 0
                        while retries < self.max_retries:
                            try:
                                x, y = self._generate_fake_batch()
                                queue.put((x, y))
                                print(f"[Worker PID {mp.current_process().pid}] Put batch {i}")
                                break
                            except Exception as e:
                                retries += 1
                                print(f"[Worker] Retry {retries} on batch {i}: {e}")
                                time.sleep(0.2)
                        else:
                            print(f"[Worker] Gave up on batch {i}")
                    pipe.send("done")
            except EOFError:
                break

        print(f"[Worker PID {mp.current_process().pid}] Exiting...")

    def _start_workers(self):
        for _ in range(self.num_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=self._worker_fn,
                args=(self.queue, child_conn, self.shutdown_event),
            )
            p.daemon = True
            p.start()
            self.workers.append(p)
            self.control_pipes.append(parent_conn)

    def run_epoch(self):
        """Trigger one epoch and return an iterable"""
        self._batches_returned = 0
        self._batches_expected = self.batches_per_epoch

        # Tell each worker to start this epoch
        per_worker = self.batches_per_epoch // self.num_workers
        for conn in self.control_pipes:
            conn.send(("run", per_worker))

        return self  # So we can use `for batch in loader.run_epoch():`

    def __iter__(self):
        return self

    def __next__(self):
        if self._batches_returned >= self._batches_expected:
            raise StopIteration

        while True:
            try:
                batch = self.queue.get(timeout=1)
                self._batches_returned += 1
                return batch
            except Empty:
                # Check if all workers are done
                all_done = all(conn.poll() and conn.recv() == "done" for conn in self.control_pipes)
                if all_done and self.queue.empty():
                    raise StopIteration

    def shutdown(self):
        self.shutdown_event.set()
        for conn in self.control_pipes:
            conn.close()
        for p in self.workers:
            p.join()
        print("✅ All workers shutdown.")

# ---------------------
# 🧪 Example Usage
# ---------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    loader = AsyncPrefetchLoader(
        num_workers=2,
        batch_size=32,
        batches_per_epoch=8,
        queue_size=4,
        persistent_workers=True,
    )

    for epoch in range(3):
        print(f"\n🌟 Epoch {epoch}")
        for i, (x, y) in enumerate(loader.run_epoch()):
            print(f"[Main] Epoch {epoch} - Batch {i} — x: {x.shape}, y: {y.shape}")
            time.sleep(0.2)

    loader.shutdown()