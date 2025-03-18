import time
import multiprocessing as mp

def worker(event):
    """Worker waits until the event is set before proceeding."""
    print("Worker: Waiting for event to be set...")
    event.wait()  # Block until event is set
    print("Worker: Event received! Processing data...")

def main():
    event = mp.Event()  # Create an event object

    # Create and start worker process
    process = mp.Process(target=worker, args=(event,))
    process.start()

    # Simulate some work in the main process
    time.sleep(2)
    print("Main: Setting event to allow worker to continue...")
    event.set()  # Signal the worker process to continue

    # Wait for worker to finish
    process.join()
    print("Main: Worker has finished execution.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Required for some OS (e.g., macOS, Windows)
    main()