from keras.datasets import mnist
# command line
# /Applications/Python\ 3.6/Install\ Certificates.command
from keras.utils import np_utils

import numpy as np
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.enable_eager_execution()

a = int(input("image number >> "))

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data("mnist.npz")

print("학습셋 이미지 수 : %d 개" %(X_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" %(X_test.shape[0]))

import matplotlib.pyplot as plt

plt.imshow(X_train[a], cmap='Greys')
plt.show()

# for x in X_train[a]:
#     for i in x:
#         sys.stdout.write('%d\t' %i)
#     sys.stdout.write('\n')

X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float64')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64')/255

print("Class: %d " %(Y_class_train[a]))

Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)

print(Y_train[a])