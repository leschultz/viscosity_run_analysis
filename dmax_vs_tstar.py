from matplotlib import pyplot as pl
import pandas as pd

df = pd.read_csv('../jobs_data/fragility.txt')
df['dmax'] = [0.2, 0.2, 5.0, 3.0, 2.0, 1.0, 1.9e-5, 10.5]

x = df['tg/tstar'].values
y = df['dmax'].values

fig, ax = pl.subplots()

groups = df.groupby('job')

for i, j in groups:

    x = j['tg/tstar'].values
    y = j['dmax'].values

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

pl.show()
print(df)
