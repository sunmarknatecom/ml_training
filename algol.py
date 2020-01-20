import time
import matplotlib.pyplot as plt
import numpy as np

def duplication_On2(lst):
    ct = 0
    for x in lst:
        if lst.count(x) > 1:
             ct += 1
    return ct

def duplication_On(lst):
    cts = {}
    for x in lst:
        if x in cts:
            ctx[x] += 1
        else:
            cts[x] = 1
    counts_above_1 = [ct for x, ct in cts.items() if ct > 1]
    return sum(counts_above_1)

def timeit(func, arg):
    start = time.time()
    func(arg)
    stop = time.time()
    return stop-start

times_On, times_On2 = [], []
ns = range(50)
for n in ns:
    lst = list(np.random.uniform(size=n))
    times_On2.append(timeit(duplication_On2, lst))
    times_On.append(timeit(duplication_On, lst))

plt.plot(times_On2, "--", label="O(n^s)")
plt.plot(times_On, label="O(n)")
plt.xlabel("Length of array")
plt.ylabel("time(sec)")
plt.title("Time to count the duplicate values")
plt.legend(loc="upper left")
plt.show()