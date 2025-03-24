from multiprocessing import Pool, Queue
import multiprocessing as mp

def worker(q):
    print(q.get())


with Pool(1) as pool:
  q = Queue()
  q.put(7)
  pool.apply(worker, args=(q,))



  