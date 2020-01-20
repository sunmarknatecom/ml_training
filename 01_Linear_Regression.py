import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값: ", mx)
# x의 평균값:  5.0
print("y의 평균값: ", my)
# y의 평균값:  90.5
divisor = sum([(mx-i)**2 for i in x])
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
            d +=(x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)
print("분      모: ", divisor)
# 분모:  20.0
print("분      자: ", dividend)
# 분자:  46.0
a = dividend / divisor
b = my - (mx*a)
print("기울기 a  = ", a)
# 기울기 a =  2.3
print("y 절편 b  = ", b)
# y 절편 b =  79.0