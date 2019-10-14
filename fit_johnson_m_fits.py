from matplotlib import pyplot as pl

from sklearn import linear_model

import pandas as pd
import numpy as np

df = pd.read_csv('../jobs_data/m_fit.txt')
dfjohnson = pd.read_csv('../dmax_johnson/johnson/data_clean.csv')

columns = [
           'Alloy',
           'Tg (K)',
           'TL (K)',
           'm',
           'dexp (mm)'
           ]

dfjohnson = dfjohnson[columns]

dfjohnson.columns = [
                     'composition',
                     'tg_exp',
                     'tl',
                     'm',
                     'dmax'
                     ]

dfjohnson = dfjohnson.dropna()

# Gather Tg/T* and Tg/Tl
dfjohnson['tg_exp/tl'] = dfjohnson['tg_exp']/dfjohnson['tl']

# Take the log of the squared dmax
df['log(dmax^2)'] = np.log10(df['dmax_mean']**2)
dfjohnson['log(dmax^2)'] = np.log10(dfjohnson['dmax']**2)

X_md_train = df[['m_pred', 'tg_md/tl_mean']].values
X_johnson_train = dfjohnson[['m', 'tg_exp/tl']].values

y_md_train = df['log(dmax^2)'].values
y_johnson_train = dfjohnson['log(dmax^2)'].values

reg = linear_model.LinearRegression()
reg.fit(X_johnson_train, y_johnson_train)

y_johnson_pred = reg.predict(X_johnson_train)

y_md_pred = reg.predict(X_md_train)

dfjohnson['log(dmax^2)_pred'] = y_johnson_pred
df['log(dmax^2)_pred'] = y_md_pred

fig, ax = pl.subplots()

ax.plot(
        y_johnson_pred,
        y_johnson_train,
        marker='.',
        linestyle='none',
        label='Original Data'
        )

ax.plot(
        y_md_pred,
        y_md_train,
        marker='8',
        linestyle='none',
        label=r'MD Data'
        )

for i, txt in enumerate(df['composition'].values):
    ax.annotate(txt, (y_md_pred[i], y_md_train[i]))

ax.legend()
ax.grid()

ax.set_xlabel(r'Predicted $log(dmax^2)$ $[log(mm)]$')
ax.set_ylabel(r'Actual $log(dmax^2)$ $[log(mm)]$')

fig.tight_layout()

fig.savefig('../jobs_plots/johnson_m_fit')

df.to_csv('../jobs_data/predictions.txt')
df.to_csv('../jobs_data/johnson_predictions.txt')

print(df)

pl.show()
