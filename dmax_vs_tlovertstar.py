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

groups = df.groupby('job')

for i, j in groups:

    x = j['TL (K)'].values/j['tstar'].values
    y = j['dexp (mm)'].values**2

    ax.plot(
            x,
            y,
            marker='*',
            linestyle='none',
            label=i
            )

ax.grid()
ax.legend()

ax.set_xlabel(r'$1000/T^{*}$')
ax.set_ylabel(r'$D_{max}^{2}$ $[mm^{2}]$')

ax.set_yscale('log')

fig.tight_layout()
fig.savefig('../jobs_plots/dmax_vs_tlovertstar')

print(df)
pl.show()
