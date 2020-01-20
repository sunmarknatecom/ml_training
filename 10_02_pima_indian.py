# pima-indians-diabetes.csv

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

ds = np.loadtxt(input("\nEx) pima-indians-diabetes.csv\nFilename ? >>"), delimiter=",")

len_label = int(np.shape(ds)[1] - 1)

X = ds[:, :len_label]
Y = ds[:, len_label]

model = Sequential()
model.add(Dense(12, input_dim=len_label, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=300, batch_size=20)

print("\n Accuracy: %02.2f %%" % (model.evaluate(X,Y)[1]*100))