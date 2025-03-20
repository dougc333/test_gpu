import torch
import torch.multiprocessing as mp

def run(rank, python_dict):
    python_dict[rank] += torch.randn(2)

def run_example(share_memory):
    print(f"share_memory={share_memory}")
    nproc=4
    python_dict = {}
    for i in range(nproc):
        python_dict[i] = torch.zeros(2)
        if share_memory:
            python_dict[i].share_memory_()
    print(f"before={python_dict}")
    processes = []
    for rank in range(nproc):
        p = mp.Process(target=run, args=(rank, python_dict,))
        processes.append(p)
    for proc in processes:
        proc.start()
        proc.join()
    print(f"after={python_dict}\n")

if __name__ == "__main__":
    run_example(share_memory=False)
    run_example(share_memory=True)
