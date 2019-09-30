from matplotlib import pyplot as pl
import pandas as pd
import numpy as np


def johnson(x, y, a, b, c):
    return a+b*x+c*y

dfjohnson = pd.read_csv('../../dmax_johnson/matches.txt')
dfjohnson['Alloy'] = [
                      'Ni400P100',
                      'Cu250Zr250',
                      'Zr300Cu150Al50'
                      ]

dfjohnson.rename(columns={'Alloy': 'composition'}, inplace=True)

df = pd.read_csv('../jobs_data/fragility.txt')
df['composition'] = df['job'].apply(lambda x: x.split('_')[1])
df = df.merge(dfjohnson, on=['composition'])

tg = df['tg'].values
tl = df['TL (K)'].values
trg = df['trg'].values
tstar = df['tstar'].values
dmax = df['dexp (mm)'].values

a = -10.36
b = 25.6
c = 0.0481

pred = johnson(tg/tl, tl/tstar, a, b, c)
print(pred)
print(tg/tl)

fig, ax = pl.subplots()

ax.plot(
        np.log10(pred**2),
        np.log10(dmax**2),
        marker='.',
        linestyle='none'
        )

ax.grid()
ax.legend()

ax.set_xlabel(r'Observed $log(d_{max}^{2})$')
ax.set_ylabel(r'Predicted $log(d_{max}^{2})$')

fig.tight_layout()
fig.savefig('../jobs_plots/tlovertstar_fragility_fit')

print(df)
pl.show()
