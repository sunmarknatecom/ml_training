import pandas as pd

ed = [pd.read_csv('/Users/sunmark/desktop/ml/datasets/shikoku_electricity_%d.csv' % year, skiprows=3, names=['DATE', 'TIME', 'consumption'], parse_dates={'date_hour':['DATE','TIME']}, index_col='date_hour') for year in [2012, 2013, 2014, 2015, 2016]]

elec_data = pd.concat(ed)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

delta = elec_data.index - pd.to_datetime('2012/07/01 00:00:00')
elec_data['time'] = delta.days + delta.seconds / 3600.0/ 24.0

plt.scatter(elec_data['time'], elec_data['consumption'], s=0.1)
plt.xlabel('days from 2012/7/1')
plt.ylabel('electricity consumption(*10000 kWh)')
# plt.savefig('7_4_1_1_graph.png')
plt.show()

plt.figure(figsize=(10,6))

plt.hist(elec_data['consumption'], bins=50, color='gray')
plt.xlabel('electricity consumption(*10000 kW)')
plt.ylabel('count')
plt.show()