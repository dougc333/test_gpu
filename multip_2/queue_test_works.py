from multiprocessing import Process, Queue

def worker(q):
    print(q.get())

if __name__ == "__main__":
  q = Queue()
  q.put(7)
  p = Process(target=worker, args=(q,))
  p.start()
  p.join()

# why does this work
#https://stackoverflow.com/questions/75233794/how-is-the-multiprocessing-queue-instance-serialized-when-passed-as-an-argument