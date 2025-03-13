import torch
import torch.multiprocessing as mp
from torch.multiprocessing import reductions
import time

# slow way copy tensor fast way use reduce to send to shared memory

def copy_worker(tensor):
    print(f"copy worker received tensor:{tensor}")

def copy_tensor():
    mp.set_start_method("spawn")
    t = torch.randn(500,500)
    queue = mp.Queue()
    queue.put(t)

    p = mp.Process(target=copy_worker, args=(queue.get(),))
    p.start()
    p.join()


def fast_worker(queue):
    tensor = queue.get()  # Receives shared tensor
    print(f"Received shared tensor: {tensor}")

def fast_tensor():
    #mp.set_start_method('spawn')

    tensor = torch.randn(500, 500)  # CPU tensor
    # Manually register tensor for shared memory
    handle = reductions.reduce_tensor(tensor)

    queue = mp.Queue()
    queue.put(handle)  # Sends shared memory handle instead of copying

    p = mp.Process(target=fast_worker, args=(queue,))
    p.start()
    p.join()

if __name__ == "__main__":

    start_time = time.monotonic()
    copy_tensor()
    print("copy tensor elapsed:",(time.monotonic()-start_time))
    start_time = time.monotonic()
    fast_tensor()
    print("fast tensor elapsed:",(time.monotonic()-start_time))
    print("this measurement isnt really correct we measuring print times")
    