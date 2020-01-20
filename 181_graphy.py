import pandas as pd

tmp = pd.read_csv(u'/Users/sunmark/desktop/ml/datasets/47891_city.csv', parse_dates={'date_hour': ["일시"]}, index_col="date_hour", na_values="x")

del tmp["시"]

columns = {"강수량(mm)": "rain", "기온(℃)": "temperature", "일조 시간(h)": "sunhour", "습도(%)": "humid",}

tmp.rename(columns=columns, inplace=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
delta = tmp.index - pd.to_datetime( '2012/07/01 00:00:00')
tmp['time']  = delta.days + delta.seconds/3600.0/24.0

plt.scatter(tmp['time'], tmp['temperature'], s=0.1)
plt.xlabel('days from 2012/7/1')
plt.ylabel('Temperature(C degree)')
plt.show()