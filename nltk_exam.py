import time, numpy as np, matplotlib.pyplot as plt

def time_numpy(n):
    a = np.arange(n)
    start = time.time()
    bigger = a + 1
    stop = time.time()
    return stop - start

def time_python(n):
    l = range(n)
    start = time.time()
    bigger = [x + 1 for x in l]
    stop = time.time()
    return stop - start

n_trials = 10
ns = range(20, 500)
ratios = []
for n in ns:
    python_total = sum([time_python(n) for _ in range(n_trials)])
    numpy_total = sum([time_numpy(n) for _ in range(n_trials)])
    ratios.append(python_total/numpy_total)

plt.plot(ns, ratios)
plt.xlabel("length of array")
plt.ylabel("ratio of python/numpy")
plt.title("comparing of woring time")
plt.show()

del n_trials, ns, ratios, python_total, numpy_total