from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np


dfkelton = pd.read_csv('../kelton/tgovertstar.csv')

dfjohnson = pd.read_csv('../dmax_johnson/matches.txt')
dfjohnson['Alloy'] = [
                      'Ni400P100',
                      'Cu250Zr250',
                      'Zr300Cu150Al50'
                      ]

dfjohnson.rename(columns={'Alloy': 'composition'}, inplace=True)
dfjohnson = dfjohnson[['composition', 'dexp (mm)', 'TL (K)', 'Tg (K)', 'm']]
dfjohnson.columns = ['composition', 'dmax', 'tl', 'tg_exp', 'm']

dfward = pd.read_csv('../dmax_ward/matches.txt')

# Composition for 500 atoms
dfward['composition'] = [
                         'Al0Cu200Zr300',
                         'Ni300Nb200',
                         'Cu250Zr250',
                         'Cu250Zr250',
                         'Al35Cu230Zr235',
                         'Al35Cu225Zr240',
                         ]

dftl = pd.read_csv('../tl/tl.txt')
dftl = dftl.dropna()

dfward = dfward.merge(dftl, on=['composition'])
dfward = dfward[['composition', 'D_max', 'tl']]
dfward.columns = ['composition', 'dmax', 'tl']

df = pd.read_csv('../jobs_data/fragility.txt')
df['composition'] = df['job'].apply(lambda x: x.split('_')[1])

dfjohnson = df.merge(dfjohnson, on=['composition'])
dfward = df.merge(dfward, on=['composition'])

df = pd.concat([dfjohnson, dfward], sort=True)
df.rename(columns={'tg': 'tg_md'}, inplace=True)
df.rename(columns={'tstar': 't*'}, inplace=True)

# Gather Tg/T* and Tg/Tl
df['tg_md/t*'] = df['tg_md']/df['t*']
df['tg_exp/t*'] = df['tg_exp']/df['t*']

df['tg_md/tl'] = df['tg_md']/df['tl']
df['tg_exp/tl'] = df['tg_exp']/df['tl']

# Take mean values for each composition
groups = df.groupby(['composition'])
mean = groups.mean().add_suffix('_mean').reset_index()
std = groups.std().add_suffix('_std').reset_index()
sem = groups.sem().add_suffix('_sem').reset_index()
count = groups.count().add_suffix('_count').reset_index()

# Merge data
df = mean.merge(sem)
df = df.merge(std)
df = df.merge(count)

X_md_train = df[['tg_md/t*_mean']].values
X_kelton_train = dfkelton[['tg/t*']].values

y_md_train = df['m_mean'].values
y_kelton_train = dfkelton['m'].values

reg = linear_model.LinearRegression()
reg.fit(X_kelton_train, y_kelton_train)

y_md_pred = reg.predict(X_md_train)

fig, ax = pl.subplots()

ax.plot(
        dfkelton['tg/t*'],
        dfkelton['m'],
        marker='.',
        linestyle='none',
        label='Kelton Papers'
        )

ax.plot(
        df['tg_md/t*_mean'],
        df['m_mean'],
        marker='8',
        linestyle='none',
        label=r'$T_{g}$ MD'
        )

ax.plot(
        df['tg_md/t*_mean'],
        y_md_pred,
        marker='8',
        linestyle='none',
        label=r'Predicted $T_{g}/T_{*}$ from fitting to Kelton'
        )

ax.legend()
ax.grid()

ax.set_xlabel(r'$T_{g}/T_{*}$ fit [-]')
ax.set_ylabel(r'Fragility Index (m)')

fig.tight_layout()

fig.savefig('../jobs_plots/m_vs_tgovertstar')

df['m_pred'] = y_md_pred

df.to_csv('../jobs_data/m_fit.txt', index=False)
print(df)

pl.show()
