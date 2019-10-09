from matplotlib import colors as mcolors
from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

# Colors for plots
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = [j for i, j in colors.items()]
colors = iter(colors)

dfjohnson = pd.read_csv('../dmax_johnson/matches.txt')
dfjohnson['Alloy'] = [
                      'Ni400P100',
                      'Cu250Zr250',
                      'Zr300Cu150Al50'
                      ]

dfjohnson.rename(columns={'Alloy': 'composition'}, inplace=True)


df = pd.read_csv('../jobs_data/fragility.txt')
df['composition'] = df['job'].apply(lambda x: x.split('_')[1])

df = df.merge(dfjohnson, on=['composition'])

columns = [
           'job',
           'composition',
           'visc',
           'tg',
           'Tg (K)',
           'TL (K)',
           'tstar',
           'm',
           'dexp (mm)'
           ]

df = df[columns]
df.columns = [
              'job',
              'composition',
              'visc',
              'tg_md',
              'tg_exp',
              'tl',
              'tstar',
              'm',
              'dmax'
              ]

dfjohnson = pd.read_csv('../dmax_johnson/johnson/data_clean.csv')

columns = [
           'Alloy',
           'Tg (K)',
           'TL (K)',
           'm',
           'dexp (mm)'
           ]

dfjohnson = dfjohnson[columns]

dfjohnson.columns = [
                     'composition',
                     'tg_exp',
                     'tl',
                     'm',
                     'dmax'
                     ]

dfjohnson = dfjohnson.dropna()

# Gather Tg/T* and Tg/Tl
df['tg_md/tstar'] = df['tg_md']/df['tstar']
df['tg_exp/tstar'] = df['tg_exp']/df['tstar']

df['tg_md/tl'] = df['tg_md']/df['tl']
df['tg_exp/tl'] = df['tg_exp']/df['tl']

dfjohnson['tg_exp/tl'] = dfjohnson['tg_exp']/dfjohnson['tl']

# Take the log of the squared dmax
df['log(dmax^2)'] = np.log10(df['dmax']**2)
dfjohnson['log(dmax^2)'] = np.log10(dfjohnson['dmax']**2)

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

X_md_train = df[['m_mean', 'tg_md/tl_mean']].values
X_exp_train = dfjohnson[['m', 'tg_exp/tl']].values

X_md_train = np.hstack((X_md_train, np.ones((X_md_train.shape[0],1))))
X_exp_train = np.hstack((X_exp_train, np.ones((X_exp_train.shape[0],1))))

y_md_train = df['log(dmax^2)_mean'].values
y_exp_train = dfjohnson['log(dmax^2)'].values

reg = linear_model.LinearRegression()
reg.fit(X_exp_train, y_exp_train)

y_exp_pred = reg.predict(X_exp_train)
y_md_pred = reg.predict(X_md_train)

fig, ax = pl.subplots()

ax.plot(
        y_exp_pred,
        y_exp_train,
        marker='.',
        linestyle='none',
        label='Data: Original'
        )

ax.plot(
        y_md_pred,
        y_md_train,
        marker='8',
        linestyle='none',
        label='Data: Substituted'
        )

ax.set_title(r'$T_{g}$ Substituted')

ax.legend()
ax.grid()

ax.set_xlabel(r'$log(dmax^2)$ fit [mm]')
ax.set_ylabel(r'$log(dmax^2)$ [mm]')

fig.tight_layout()

fig.savefig('../jobs_plots/johnson_tg_substitution')

print(df)

pl.show()
