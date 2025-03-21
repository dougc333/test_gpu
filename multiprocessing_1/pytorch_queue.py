import torch
import torch.multiprocessing as mp
import time

def producer(queue, num_tensors):
    """Generates tensors and puts them into the queue"""
    for i in range(num_tensors):
        tensor = torch.randn(3, 3)  # Generate a random tensor
        print(f"Producer: Produced tensor {i}\n{tensor}")
        queue.put(tensor)  # Send tensor to queue
        time.sleep(1)  # Simulate work
    
    queue.put(None)  # Send termination signal
    print("Producer: Finished producing.")

def consumer(queue):
    """Consumes tensors from the queue and processes them"""
    while True:
        tensor = queue.get()  # Receive tensor from queue
        if tensor is None:  # Stop when termination signal is received
            break
        print(f"Consumer: Processing tensor\n{tensor * 2}")  # Simulate processing

    print("Consumer: Finished consuming.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Required for PyTorch multiprocessing

    queue = mp.Queue()  # Create a PyTorch multiprocessing queue

    producer_process = mp.Process(target=producer, args=(queue, 5))
    consumer_process = mp.Process(target=consumer, args=(queue,))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

    print("Main Process: Execution Completed")
