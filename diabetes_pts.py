import sklearn.datasets
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score

diabetes = sklearn.datasets.load_diabetes()
X, Y = normalize(diabetes['data']), diabetes['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.8)
linear = LinearRegression()
linear.fit(X_train, Y_train)
preds_linear = linear.predict(X_test)
corr_linear = round(pd.Series(preds_linear).corr(pd.Series(Y_test)), 3)
rsquared_linear = r2_score(Y_test, preds_linear)

print("Linear coefficient:")
print(linear.coef_)
plt.scatter(preds_linear, Y_test)
plt.title("Linear Regression Results, Corr = %f R^2 score = %f" %(corr_linear, rsquared_linear))
plt.xlabel("Preds value")
plt.ylabel("True value")
plt.plot(Y_test, Y_test, 'k--')
plt.show()

lasso = Lasso()
lasso.fit(X_train, Y_train)
preds_lasso = lasso.predict(X_test)
corr_lasso = round(pd.Series(preds_lasso).corr(pd.Series(Y_test)), 3)
rsquared_lasso = round(r2_score(Y_test, preds_lasso), 3)

print("Lasso coefficient:")
print(lasso.coef_)
plt.scatter(preds_lasso, Y_test)
plt.title("Lasso regression results. corr= %f R^2 score = %f" %(corr_lasso, rsquared_lasso))
plt.xlabel("Preds value")
plt.ylabel("True value")
plt.plot(Y_test, Y_test, 'k--')
plt.show()