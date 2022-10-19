from multiprocessing import Queue
P: "Queue[Path]" = Queue()

def myfunc(q: Queue[int]):
    print("worked")

q = Queue()

myfunc(q)
