from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

ds = np.loadtxt("pimaindian.csv", delimiter=",")

X = ds[:,0:8]
Y = ds[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=200, batch_size=10)

print("\n Accuracy: %.4f" %(model.evaluate(X,Y)[1]))
