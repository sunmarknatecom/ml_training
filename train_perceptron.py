import numpy as np                                      # 1 module
from sklearn import datasets                            # 2 module
from sklearn.model_selection import train_test_split    # 3 module
from sklearn.preprocessing import StandardScaler        # 4 module
from sklearn.linear_model import Perceptron             # 5 module
from sklearn.metrics import accuracy_score              # 6 module

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print('클래스 레이블:', np.unique(y))

# 3 module need
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('y의 레이블 카운트:', np.bincount(y))
print('y_train의 레이블 카운트:', np.bincount(y_train))
print('y_test의 레이블 카운트:', np.bincount(y_test))

# 4 module need
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 5 module need
ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수: %d' %(y_test != y_pred).sum())

# 6 module need
print('정확도: %.2f' %accuracy_score(y_test, y_pred))
print('정확도: %.2f' %ppn.score(X_test_std, y_test))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_deicion_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 마커와 컬러맵을 설정합니다.
    markers = ('s', 'x', 'o', '^')