import torch.multiprocessing as mp

# Function for first process
def print_hello():
    print("Hello from Process 1!")

# Function for second process
def print_world():
    print("Hello from Process 2!")

def main():
    mp.set_start_method("spawn", force=True)  # Required for multiprocessing on some OS

    # Create two separate processes
    process1 = mp.Process(target=print_hello)
    process2 = mp.Process(target=print_world)

    # Start both processes
    process1.start()
    process2.start()

    # Wait for both processes to complete
    process1.join()
    process2.join()

if __name__ == "__main__":
    main()