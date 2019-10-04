from matplotlib import colors as mcolors
from matplotlib import pyplot as pl
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import re


def johnson(x, y, a, b, c):
    return a+b*x+c*y


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

x = dfj['tl/tstar'].values
y = dfj['m'].values

degree = 2
coeffs = np.polyfit(x, y, degree)
fit = np.poly1d(coeffs)

xfit = np.linspace(min(x), max(x))
yfitpoly = fit(xfit)

dfw['m'] = fit(dfw['tl/tstar'].values)

groups = dfj.groupby(['composition'])

sig = 6
coeffsstr = str([str(i)[:sig] for i in coeffs])

fig, ax = pl.subplots()
ax.plot(xfit, yfitpoly, label='Polyfit Coefficients '+coeffsstr)

for i, j in groups:

    x = j['tl/tstar'].values
    y = j['m'].values

    ax.plot(
            x,
            y,
            marker='8',
            linestyle='none',
            label=i
            )

groups = dfw.groupby(['composition'])

for i, j in groups:

    x = j['tl/tstar'].values
    y = j['m'].values

    ax.plot(
            x,
            y,
            marker='8',
            linestyle='none',
            label=i
            )

ax.grid()
ax.legend()

ax.set_xlabel(r'$T_{l}/T^{*}$')
ax.set_ylabel(r'$m$')

ax.set_yscale('log')

fig.tight_layout()
fig.savefig('../jobs_plots/m_vs_tlovertstar')

print(dfj)
print(dfw)

groups = dfw.groupby(['composition'])
meanw = groups.mean()

dfw = meanw

a = -10.36
b = 25.6
c = 0.0481

df = pd.read_csv('../../dmax_johnson/johnson/data_clean.csv')

df = df[['Alloy', 'Tg (K)', 'TL (K)', 'm', 'dexp (mm)', 'dcalc (mm)']]
df.columns = ['composition', 'tg', 'tl', 'm', 'dmax', 'dcalc']
df = df.dropna()

df['trg'] = df['tg']/df['tl']

X_train = df[['trg', 'm']].values
X_train = np.hstack((X_train, np.ones((X_train.shape[0],1))))
y_train = np.log10(df['dmax'].values**2)

w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), X_train.T), y_train)
a, b, c = w

dfw['trg'] = dfw['tg']/dfw['tl']
X_test = dfw[['trg', 'm']].values
X_test = np.hstack((X_test, np.ones((X_test.shape[0],1))))

yw_original = np.log10(dfw['dmax']**2)
yw_pred = np.matmul(X_test, w)

df['log(dmax^2)'] = np.log10(df['dmax']**2)
df['log(dmax^2)_fit'] = a*df['trg']+b*df['m']+c

x = df['log(dmax^2)_fit'].values
y = df['log(dmax^2)'].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

fig, ax = pl.subplots()

ax.plot(x, y, marker='.', linestyle='none', label=r'[$\Lambda_{t}$, $\Lambda_{m}$, $log_{10}(dmax_{0}^{2})$]='+str(w)+'\n'+r'$r=$'+str(r_value))

ax.plot(yw_pred, yw_original, marker='8', linestyle='none', label='Ward')


ax.legend()
ax.grid()

ax.set_xlabel(r'$log(dmax^2)$ fit [mm]')
ax.set_ylabel(r'$log(dmax^2)$ [mm]')

pl.show()
print(df)
