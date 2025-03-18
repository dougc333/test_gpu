import torch
import torch.multiprocessing as mp
import torch.profiler

# Define worker function for sending data
def sender(queue, data):
    with torch.profiler.record_function("send_data"):
        queue.put(data)  # Send tensor to queue

# Define worker function for receiving data
def receiver(queue):
    with torch.profiler.record_function("receive_data"):
        data = queue.get()  # Receive tensor from queue
        return data

def main():
    # Set start method (important for Windows & MacOS)
    mp.set_start_method("spawn", force=True)

    # Create a multiprocessing queue
    queue = mp.Queue()

    # Example tensor to send
    tensor_data = torch.randn(1000, 1000)  # Large tensor for profiling

    # Enable PyTorch Profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Create processes
        sender_process = mp.Process(target=sender, args=(queue, tensor_data))
        receiver_process = mp.Process(target=receiver, args=(queue,))

        # Start processes
        sender_process.start()
        receiver_process.start()

        # Wait for processes to finish
        sender_process.join()
        receiver_process.join()

    # Print profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total"))

if __name__ == "__main__":
    main()