import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size, filepath):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{filepath}",
        world_size=world_size,
        rank=rank,
    )

    tensor = torch.zeros(1).share_memory_()
    tensor += rank + 1
    print(f"[Rank {rank}] Local tensor: {tensor}")

    dist.barrier()
    if rank == 0:
        print("[Rank 0] Final shared tensor (from rank 0):", tensor)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    filepath = "/tmp/ddp_sync_file"
    if os.path.exists(filepath):
        os.remove(filepath)

    mp.spawn(run, args=(world_size, filepath), nprocs=world_size)