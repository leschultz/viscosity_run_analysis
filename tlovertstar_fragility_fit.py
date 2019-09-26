from matplotlib import pyplot as pl
import pandas as pd
import numpy as np

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

fig, ax = pl.subplots()

groups = df.groupby('composition')

for i, j in groups:

    x = j['TL (K)'].values/j['tstar'].values
    y = j['m'].values

    ax.plot(
            x,
            y,
            marker='*',
            linestyle='none',
            label=i
            )

x = df['TL (K)'].values/df['tstar'].values
y = df['m'].values

degree=3
coefs = np.polyfit(x, y, degree)
ffit = np.poly1d(coefs)

xfit = np.linspace(min(x), max(x))
yfit = ffit(xfit)

ax.plot(
        xfit,
        yfit,
        linestyle=':',
        color='k',
        label='Polynomial Fit Degree='+str(degree)
        )

ax.grid()
ax.legend()

ax.set_xlabel(r'$T_{l}/T^{*}$')
ax.set_ylabel(r'm')

fig.tight_layout()
fig.savefig('../jobs_plots/tlovertstar_fragility_fit')

print(df)
pl.show()
