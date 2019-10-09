from matplotlib import colors as mcolors
from matplotlib import pyplot as pl
import pandas as pd
import numpy as np

# Colors for plots
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = [j for i, j in colors.items()]
colors = iter(colors)

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

columns = [
           'job',
           'composition',
           'visc',
           'tg',
           'TL (K)',
           'tstar',
           'm',
           'dexp (mm)'
           ]

df = df[columns]
df.columns = ['job', 'composition', 'visc', 'tg', 'tl', 'tstar', 'm', 'dmax']
print(df)

# Gather Tg/T* and Tg/Tl
df['tg/tstar'] = df['tg'].values/df['tstar'].values
df['trg'] = df['tg'].values/df['tl'].values

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

    x = j['tg/tstar_mean'].values
    y = j['dmax_mean'].values**2
    xstd = j['tg/tstar_std'].values
    xsem = j['tg/tstar_sem'].values

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

ax.set_xlabel(r'$T_{g}/T_{*}$')
ax.set_ylabel(r'$D_{max}^{2}$ $[mm^{2}]$')

ax.set_yscale('log')

fig.tight_layout()
fig.savefig('../jobs_plots/dmax_vs_tgovertstar')

print(df)
pl.show()
