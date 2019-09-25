from matplotlib import pyplot as pl
import pandas as pd
import numpy as np

dfjohnson = pd.read_csv('../../dmax_johnson/matches.txt')
dfjohnson['Alloy'] = [
                      'viscosity_Ni400P100_dT50K',
                      'viscosity_Cu250Zr250_dT50K',
                      'viscosity_Zr300Cu150Al50_dT50K'
                      ]

dfjohnson.rename(columns={'Alloy': 'job'}, inplace=True)

df = pd.read_csv('../jobs_data/fragility.txt')
df = df.merge(dfjohnson, on=['job'])

fig, ax = pl.subplots()

groups = df.groupby('job')

for i, j in groups:

    x = j['tg/tstar'].values
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

ax.set_xlabel(r'$T_{g}/T^{*}$')
ax.set_ylabel(r'$D_{max}^{2}$ $[mm^{2}]$')

ax.set_yscale('log')

fig.tight_layout()
fig.savefig('../jobs_plots/dmax_vs_tgovertstar')

print(df)
pl.show()
