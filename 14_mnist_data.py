from keras.utils import np_utils
from brew_load_mnist import f_load_mnist

import numpy as np
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

X_train, Y_class_train = f_load_mnist('', kind='train')
X_test, Y_class_test = f_load_mnist('', kind='t10k')

# (X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()

print("학습셋 이미지 수: %d 개" % (X_train.shape[0]))
print("테스트셋 이미지 수: %d 개" % (X_test.shape[0]))

import matplotlib.pyplot as plt

plt.imshow(X_train[0], cmap='Greys')
plt.show()