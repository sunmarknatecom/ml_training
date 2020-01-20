import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target-self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        return np.dot(X, self.w_[1:])+self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)



# iris['data']          [4.8, 3, 1.4, 0.3]
# iris['target']        [0 0 0 ~~~~ 2 2 2]
# iris['target_names']  ['setosa', 'vesicolor', 'virginica']
# iris['feature_names'] ['sepal length(cm)', 'septal width(cm)', 'petal length(cm)', 'petal width(cm)']

iris = load_iris()

X = iris['data'][0:100,[0,2]]
Y = iris['target'][0:100]
y = np.where(Y==0, -1, 1)

def plot_decison_regions(X, y, classifier, resolution=0.02):
    markers = ['s', 'x', 'o', '^', 'v']
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8, c=colors[idx],marker=markers[idx], label=cl, edgecolor='black')

def scatter_plot():
    plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

def errors_graph():
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of errors')
    plt.show()

def dcr_plot():
    global X
    global y
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plot_decison_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

if __name__=='__main__':
    print("What is your selection? \n 1. Scatter plot \n 2. Error graph \n 3. Decision Plot\n 4. All \n")
    sel_num = int(input(" "))
    if sel_num == 1:
        scatter_plot()
    elif sel_num == 2:
        errors_graph()
    elif sel_num == 3:
        dcr_plot()
    elif sel_num == 4:
        scatter_plot()
        errors_graph()
        dcr_plot()
    else:
        print("break!")
    print(" Your selection is %d." %sel_num)
