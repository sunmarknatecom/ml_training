from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv('sonar.csv', header=None)

ds = df.values
X = ds[:,0:60]
Y_obj = ds[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X,Y, epochs=200, batch_size=5)

print("\n Accuracy: %.4f" %(model.evaluate(X,Y)[1]))