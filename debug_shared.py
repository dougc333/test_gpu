import torch
import torch.multiprocessing as mp




def debug():
  device = torch.device("cpu")
  t = torch.ones(3, device=device)
  s0 = t.untyped_storage()
  print(s0.data_ptr())


if __name__ == "__main__":
  debug()