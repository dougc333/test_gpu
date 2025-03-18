import torch
import torch.profiler

def foo():
  print(torch.cuda.is_available())

def a():
  '''
  simple cuda gpu stuff
  '''
  # Profile execution
  with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
      record_shapes=True,
      profile_memory=True
  ) as profiler:
    model_operations()
    
  print(profiler.key_averages().table(sort_by="cuda_time_total"))




def model_operations():
    A = torch.randn(1000, 1000, device='cuda')
    B = torch.randn(1000, 1000, device='cuda')
    
    C = torch.matmul(A, B)  # Matrix multiplication (compute-heavy)
    D = torch.exp(A)  # Element-wise operation

    return C, D

def status():
  
  print(torch.cuda.memory_summary())

if __name__=="__main__":
  foo()
  a()
