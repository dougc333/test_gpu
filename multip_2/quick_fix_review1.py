import multiprocessing as mp
import torch





def main():
  mp.set_start_method("spawn", force=True)
  mp.set_sharing_strategy("file_system")

  q = mp.Queue()




if __name__ == '__main__':
  main()