#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sqlite3

#%%
cnx = sqlite3.connect('data.db')

#%%
df = pd.read_sql_query("SELECT transaction_date, year, sale_amt FROM housing WHERE prop_state='PA'", cnx)
print(df)


#%%
df.loc[df.year > 19, 'year'] = df.loc[df.year > 19, 'year'] + 1900

#%%
grouped = df.groupby(['year'])
averaged = grouped.aggregate({"sale_amt":np.mean})
print(averaged)
averaged.plot()

#%%
df.transaction_date = pd.to_datetime(df.transaction_date, format='%m/%d/%y')
df['YEAR'] = df.transaction_date.dt.year
df.loc[df.YEAR > 2019, 'YEAR'] = df.loc[df.YEAR > 2019, 'YEAR'] - 100

#%%
grouped = df.groupby(['YEAR'])
averaged = grouped.aggregate({"sale_amt":np.mean})
print(averaged)
averaged.plot()

#%%
