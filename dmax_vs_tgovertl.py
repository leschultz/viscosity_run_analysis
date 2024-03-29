from matplotlib import colors as mcolors
from matplotlib import pyplot as pl
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
dfjohnson = dfjohnson[['composition', 'dexp (mm)', 'TL (K)']]
dfjohnson.columns = ['composition', 'dmax', 'tl']

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

df = pd.concat([dfjohnson, dfward])

# Gather Tl/T*
df['tg/tl'] = df['tg'].values/df['tl'].values

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

groups = df.groupby(['composition'])

fig, ax = pl.subplots()
for i, j in groups:

    x = j['tg/tl_mean'].values
    y = j['dmax_mean'].values**2
    xstd = j['tg/tl_std'].values
    xsem = j['tg/tl_sem'].values

    ax.errorbar(
                x,
                y,
                xerr=xstd,
                ecolor='y',
                marker='8',
                linestyle='none',
                )

    ax.errorbar(
                x,
                y,
                xerr=xsem,
                ecolor='r',
                marker='8',
                linestyle='none',
                color=next(colors),
                label=i
                )


ax.grid()
ax.legend()

ax.set_xlabel(r'$T_{g}/T_{l}$')
ax.set_ylabel(r'$D_{max}^{2}$ $[mm^{2}]$')

ax.set_yscale('log')

fig.tight_layout()
fig.savefig('../jobs_plots/dmax_vs_tgovertl')

print(df)
pl.show()
