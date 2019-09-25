from matplotlib import pyplot as pl
import pandas as pd
import numpy as np

# Fitting parameters
loddmaxsquared = -10.36
t = 25.6
m = 0.0481

dfdmax = pd.read_csv('../../dmax_johnson/matches.txt')
print(dfdmax)

df = pd.read_csv('../jobs_data/fragility.txt')
df['dmax'] = [4.3e-2, 1.9e-5, 10.8]

x = df['tg/tstar'].values
y = df['dmax'].values

fig, ax = pl.subplots()

groups = df.groupby('job')

for i, j in groups:

    x = j['tg/tstar'].values
    y = j['dmax'].values
    y = np.log10(y**2.0)

    ax.plot(
            x,
            y,
            marker='.',
            linestyle='none',
            label=i
            )

ax.grid()
ax.legend()

ax.set_xlabel(r'$T_{g}/T^{*}$')
ax.set_ylabel(r'$D_{max}$ [mm]')

fig.tight_layout()
fig.savefig('../jobs_plots/dmax_vs_tgovertstar')

print(df)
pl.show()
