# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import statsmodels.api as sm
from pylab import rcParams
import itertools

# %%
cnx = sqlite3.connect('data.db')

# %%
state = "'MA'"
df = pd.read_sql_query(
    "SELECT transaction_date, year, month, sale_amt FROM housing WHERE prop_state="+state, cnx)
print(df)


# %%
df.transaction_date = pd.to_datetime(df.transaction_date, format='%m/%d/%y')
df['YEAR'] = df.transaction_date.dt.year
df.loc[df.YEAR > 2019, 'YEAR'] = df.loc[df.YEAR > 2019, 'YEAR'] - 100

# %% Group original
grouped = df.groupby(['YEAR', 'month'])
print(grouped)


# %% Aggregate and plot
averaged = grouped.aggregate({"sale_amt": np.mean}).reset_index()
averaged['day'] = '01'

date_t = pd.to_datetime(
    averaged[['YEAR', 'month', 'day']]).reset_index(name='date')
date_t['sale'] = averaged.sale_amt.values
date_t = date_t[date_t.date > '2000-01-01']

date_t = date_t.drop(columns=['index']).set_index('date')
date_t.plot()

# %%
date_t = date_t.resample('MS').interpolate()
print(date_t)

# %%
decomposition = sm.tsa.seasonal_decompose(date_t, model='additive')
fig = decomposition.plot()
plt.show()

# %%
if state == "'RI'":
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
elif state == "'PA'":
    order = (1, 1, 1)
    seasonal_order = (0, 1, 1, 12)
elif state == "'MA'":
    order = (0, 1, 1)
    seasonal_order = (0, 1, 1, 12)

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in list(itertools.product(p, d, q))]

mod = sm.tsa.statespace.SARIMAX(date_t, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False, enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

# %%
results.plot_diagnostics()
plt.show()

# %%
pred = results.get_prediction(
    start=pd.to_datetime('2013-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = date_t['2011':].plot(label='observed')
pred.predicted_mean.plot(
    ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()

plt.show()

# %%
