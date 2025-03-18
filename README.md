# test_gpu
t4 gpus

python multiprocessing.Pool works with Python types and limited range of objects

pytorch multiprocessing.Pool works with tensors and GPUs. Cant transfer GPU tensors between processes wo it. 
Garbage collection issues with multiprocessing.Pool vs. pytorch.multiprocessing.Pool

Pytorch multiprocessing spawn() different than python spawn()

Pytorch Pool/spawn uses pytorch SimpleQueue./ HOlds additional metadata. 
