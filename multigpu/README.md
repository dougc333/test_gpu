
#Multigpu and Multinode

Pytorch and torchrun work together. Elastic in torchrun
Multigpu: multigpu on single machine
Multinode: multiple machines each with GPU/s

Multinode requres --rdzv-id, --rdzv-backend, --rdzv-endpoint


What is difference between pytorch and Ray? 


pyton ddp.py 50 10

torchrun --standalone --nproc_per_node=2 multigpu_torchrun.py 50 10 




torchrun --nproc_per_node=1 --node_rank=0 --nodes=2 --rdzv_backend=c10d --rdzv_endpoint=xx.xx.xx.x:77777 --rdzv_id=456(any #) multinode.py 50 10 

torchrun --nproc_per_node=1 --node_rank=1 --nodes=2 --rdzv_backend=c10d --rdzv_endpoint=xx.xx.xx.x:77777 --rdzv_id=456(any #) multinode.py 50 10 

only change node rank everything else sme
inbound rules on security group


slurm:
headnode is separate? use a t2 instance? 



minGPT repo karpathy

https://www.youtube.com/watch?v=3XUG7cjte2U
