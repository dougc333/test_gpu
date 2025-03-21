import torch.multiprocessing as mp
import torch

def sender(queue):
    tensor = torch.tensor([1, 2, 3, 4, 5])  # Create a tensor
    print(f"Sender: Sending tensor {tensor}")
    queue.put(tensor)  # Put tensor into queue

# Function for the receiver process
def receiver(queue):
    received_tensor = queue.get()  # Get tensor from queue
    print(f"Receiver: Received tensor {received_tensor}")


def main():
    mp.set_start_method("spawn", force=True)  # Required for multiprocessing on some OS
    queue = mp.Queue()
    # Create two separate processes
    process1 = mp.Process(target=sender, args=(queue,))
    process2 = mp.Process(target=receiver, args=(queue,))

    # Start both processes
    process1.start()
    process2.start()

    # Wait for both processes to complete
    process1.join()
    process2.join()

if __name__ == "__main__":
    main()