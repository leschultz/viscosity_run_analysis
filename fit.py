from matplotlib import colors as mcolors
from matplotlib import pyplot as pl
from scipy import stats

import pandas as pd
import numpy as np
import re


# Colors for plots
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = [j for i, j in colors.items()]
colors = iter(colors)

dfward = pd.read_csv('../../bmg_ward/matches.txt')
dfward = dfward[['composition', 'D_max']]
dfward.columns = ['composition', 'dmax']

dfward['composition'] = [
                         'Al0Cu200Zr300',
                         'Ni300Nb200',
                         'Cu250Zr250',
                         'Cu250Zr250',
                         'Cu230Zr235Al35',
                         'Zr240Cu225Al35',
                         ]

dfwardtl = pd.read_csv('../tl/tl.txt')
dfwardtl = dfwardtl[['composition', 'tl']]
dfwardtl = dfwardtl.dropna()

dfward = dfward.merge(dfwardtl, on=['composition'])

dfjohnson = pd.read_csv('../../dmax_johnson/matches.txt')
dfjohnson['Alloy'] = [
                      'Ni400P100',
                      'Cu250Zr250',
                      'Zr300Cu150Al50'
                      ]

dfjohnson.rename(columns={'Alloy': 'composition'}, inplace=True)
dfjohnson = dfjohnson[['composition', 'TL (K)', 'm', 'dexp (mm)']]
dfjohnson.columns = ['composition', 'tl', 'm', 'dmax']

df = pd.read_csv('../jobs_data/fragility.txt')
df['composition'] = df['job'].apply(lambda x: x.split('_')[1])

dfj = df.merge(dfjohnson, on=['composition'])
dfw = df.merge(dfward, on=['composition'])

# Gather Tl/T*
dfj['tl/tstar'] = dfj['tl'].values/dfj['tstar'].values
dfw['tl/tstar'] = dfw['tl'].values/dfw['tstar'].values

dfj['trg'] = dfj['tg'].values/dfj['tl'].values
dfw['trg'] = dfw['tg'].values/dfw['tl'].values

print(dfj)
print(dfw)

dfj = dfj.groupby(['composition']).mean()
dfj['composition'] = dfj.index

dfw = dfw.groupby(['composition']).mean()
dfw['composition'] = dfw.index

print(dfj)
print(dfw)

columns = ['composition', 'trg', 'tl/tstar', 'dmax']
dfj = dfj[columns]
dfw = dfw[columns]

data = dfj.append(dfw)
data = data.dropna()

print(data)

X_train = data[['trg', 'tl/tstar']].values
X_train = np.hstack((X_train, np.ones((X_train.shape[0],1))))
y_train = np.log10(data['dmax'].values**2)

w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), X_train.T), y_train)
a, b, c = w

data['log(dmax^2)'] = np.log10(data['dmax']**2)
data['log(dmax^2)_fit'] = a*data['trg']+b*data['tl/tstar']+c

x = data['log(dmax^2)_fit'].values
y = data['log(dmax^2)'].values

print(y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

fig, ax = pl.subplots()

ax.plot(x, y, marker='.', linestyle='none', label=r'[$\Lambda_{t}$, $\Lambda_{m}$, $log_{10}(dmax_{0}^{2})$]='+str(w)+'\n'+r'$r=$'+str(r_value))

ax.legend()
ax.grid()

ax.set_xlabel(r'$log(dmax^2)$ fit [mm]')
ax.set_ylabel(r'$log(dmax^2)$ [mm]')

for i, txt in enumerate(data['composition']):
    ax.annotate(txt, (x[i], y[i]))

pl.show()
