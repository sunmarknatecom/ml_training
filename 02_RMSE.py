import numpy as np

ab = [3, 76]
data = [[2,81],[4,93],[6,91],[8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

def predict(x):
    return ab[0]*x +ab[1]

def rmse(p, a):
    return np.sqrt(((p-a) **2).mean())

def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))

predict_result = []

for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간 = %.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))

# 공부한 시간 = 2, 실제점수=81, 예측점수=82
# 공부한 시간 = 4, 실제점수=93, 예측점수=88
# 공부한 시간 = 6, 실제점수=91, 예측점수=94
# 공부한 시간 = 8, 실제점수=97, 예측점수=100
print("rmse 최종값: " + str(rmse_val(predict_result, y)))
# rmse 최종값: 3.3166247903554
