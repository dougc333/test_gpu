import multiprocessing as mp
import time

def producer(queue, num_items):
    """Produces integers and puts them into the queue"""
    for i in range(1, num_items + 1):
        print(f"Producer: Produced {i}")
        queue.put(i)  # Send integer to queue
        time.sleep(1)  # Simulate work
    
    queue.put(None)  # Send termination signal
    print("Producer: Finished producing.")

def consumer(queue):
    """Consumes integers from the queue"""
    while True:
        item = queue.get()  # Receive integer from queue
        if item is None:  # Stop when termination signal is received
            break
        print(f"Consumer: Processed {item * 2}")  # Simulate processing (doubling the value)

    print("Consumer: Finished consuming.")

if __name__ == "__main__":
    queue = mp.Queue()  # Create a multiprocessing-safe queue

    producer_process = mp.Process(target=producer, args=(queue, 5))  # Produces 5 integers
    consumer_process = mp.Process(target=consumer, args=(queue,))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

    print("Main Process: Execution Completed")
