# GPU configurations
1) aws t4 g4dn.xlarge
    - use the spot instance pricing .18 spot vs .53 on demand. Paperspace, hyperbolic, nothing comes close to spot pricing.
    - test programs for pytorch profiling, pytorch multiprocessing, pytorch distributed
    - pytorch multiprocessing contains a CUDA refcount adjustment to prevent GD allowing faster transfers via shared memory queue between processes. All transfers between python processes are pkl serialized but shared queue avoids this bottleneck. For sharing tensors between processes. pytorch multiprocsssing needed for pytorch distributed. 
3) Docker container
4) gitpod

