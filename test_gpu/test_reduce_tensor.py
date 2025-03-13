import torch
import torch.multiprocessing as mp
from torch.multiprocessing import reductions

def worker(queue):
    tensor = queue.get()  # Receives shared tensor
    print(f"Received shared tensor: {tensor}")

if __name__ == "__main__":
    mp.set_start_method('spawn')

    tensor = torch.randn(5, 5)  # CPU tensor

    # Manually register tensor for shared memory
    handle = reductions.reduce_tensor(tensor)

    queue = mp.Queue()
    queue.put(handle)  # Sends shared memory handle instead of copying

    p = mp.Process(target=worker, args=(queue,))
    p.start()
    p.join()