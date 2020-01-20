import pandas as pd

ed = [pd.read_csv('/datasets/shikoku_electricity_%d.csv' % year, skiprows=3, names=['DATE', 'TIME', 'consumption'], parse_dates={'date_hour': ['DATE', 'TIME']}, index_col='date_hour') for year in [2012, 2013, 2014, 2015, 2016]]

elec_data = pd.concat(ed)

tmp = pd.read_csv(u'/datasets/47891_city.csv', parse_dates={'date_hour':["시간"]}, index_col="date_hour", na_values="x")

del tmpe["시"]

columns = {"강수량(mm)": "rain","기온(℃)": "temperature","일조시간(h)": "sunhour","습도(％)": "humid",}

tmp.rename(columns=columns, inplace=True)

takamatsu = elec_data.join(tmp["temperature"]).dropna().as_matrix()

takamatsu_elec = takamatsu[:,0:1]
takamatsu_wthr = takamatsu[:,1:]

plt.xlabel('Temperature(C degree)')
plt.ylabel('electricity consumption(*10000 kW)')

plt.scatter(takamatsu_whhr, takamatsu_elec, s=0.5, color="gray", label='electricity consumption(measured)')

plt.show()