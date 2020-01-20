import numpy as np
import pandas as pd
import adalineGD as adaGD
import matplotlib.pyplot as plt


df = pd.read_csv('iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1) # y값이 Iris_setosa이면 -1, 아니면 1
X = df.iloc[0:100, [0,2]].values

fit, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
ada1 = adaGD.AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - learning rate 0.01')

ada2 = adaGD.AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - learning rate 0.0001')

plt.show()