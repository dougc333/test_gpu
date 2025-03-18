import torch
import torch.multiprocessing as mp
import torch.profiler

# Function to insert tensors into SimpleQueue
def producer(queue, num_tensors):
    with torch.profiler.record_function("queue_insert"):
        for _ in range(num_tensors):
            tensor = torch.randn(1000, 1000)  # Large tensor to simulate workload
            queue.put(tensor)

def main():
    mp.set_start_method("spawn", force=True)  # Required for multiprocessing on some OS
    print("in main")
    
    queue = mp.Queue()  # Create a SimpleQueue
    num_tensors = 10  # Number of tensors to insert
    print("num tensors:",num_tensors)
    print("queue:",queue)
    # Enable PyTorch Profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # Create a producer process
        producer_process = mp.Process(target=producer, args=(queue, num_tensors))
        print("producer process:",producer_process)
        # Start process
        producer_process.start()

        # Wait for completion
        producer_process.join()

    # Print profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

if __name__ == "__main__":
    print("cc")
    main()