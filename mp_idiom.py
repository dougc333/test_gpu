# mp.Pool is different than torch.multiprocessing.Pool


import torch.multiprocessing as mp

def my_sleep(i):
  import time
  print(f"sleep {i} start",i)
  time.sleep(i)
  return i
def main():
  with mp.Pool(processes =2) as pool:
    results = pool.map(my_sleep, [1,2,3,4,5])
    print("pool done results:",results)
    print("pool.map blocks unitl all processes are done no need for join")
if __name__ == "__main__":
  main()