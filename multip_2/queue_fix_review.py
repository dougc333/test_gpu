import multiprocessing as mp


def worker(q):
  print(q.get())


def main():
  mp.set_start_method("spawn", force=True)

  q = mp.Queue()
  q.put(1)
  p = mp.Process(target=worker, args=(q,))
  p.start()
  p.join()

if __name__ == "__main__":
  main()