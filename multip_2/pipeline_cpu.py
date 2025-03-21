import multiprocessing as mp
import time
import random

def data_loader(shared_data, lock, event):
    """Simulates a data loading process"""
    for i in range(5):
        time.sleep(random.uniform(0.5, 1.5))  # Simulate loading time
        with lock:
            shared_data.value = i  # Put batch ID as an integer
            print(f"[Loader] Prepared batch_{i}")
        event.set()  # Signal that data is ready

def trainer(shared_data, lock, event):
    """Simulates a training process"""
    for _ in range(5):
        print("[Trainer] Waiting for data...")
        event.wait()  # Wait for signal from loader
        with lock:
            batch = shared_data.value
            print(f"[Trainer] Training on batch_{batch}")
        event.clear()  # Reset the event

if __name__ == "__main__":
    # Shared memory for passing data
    shared_data = mp.Value('i', 0)  # 'i' = signed int

    # Lock and Event for sync
    lock = mp.Lock()
    event = mp.Event()

    # Create and start processes
    loader = mp.Process(target=data_loader, args=(shared_data, lock, event))
    trainer_proc = mp.Process(target=trainer, args=(shared_data, lock, event))

    loader.start()
    trainer_proc.start()

    loader.join()
    trainer_proc.join()

    print("✅ Multiprocessing training complete!")