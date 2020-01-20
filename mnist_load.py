import pickle, gzip, numpy, urllib.request, json


# a = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
# b = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
# c = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
# d = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

# urllib.request.urlretrieve(a)
with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    pickle.load(f, encoding='latin1')