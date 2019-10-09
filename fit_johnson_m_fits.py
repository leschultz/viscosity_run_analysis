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
X_exp_train = df[['m_mean', 'tg_exp/tl_mean']].values
X_johnson_train = dfjohnson[['m', 'tg_exp/tl']].values

X_md_train = np.hstack((X_md_train, np.ones((X_md_train.shape[0],1))))
X_exp_train = np.hstack((X_exp_train, np.ones((X_exp_train.shape[0],1))))
X_johnson_train = np.hstack((X_johnson_train, np.ones((X_johnson_train.shape[0],1))))

y_md_train = df['log(dmax^2)'].values
y_exp_train = df['log(dmax^2)'].values
y_johnson_train = dfjohnson['log(dmax^2)'].values

reg = linear_model.LinearRegression()
reg.fit(X_johnson_train, y_johnson_train)

y_johnson_pred = reg.predict(X_johnson_train)

y_md_pred = reg.predict(X_md_train)
y_exp_pred = reg.predict(X_exp_train)

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
        label=r'Substituted $T_{g}$ MD'
        )

ax.plot(
        y_exp_pred,
        y_exp_train,
        marker='8',
        linestyle='none',
        label=r'Original Experimental Points'
        )

ax.set_title(r'$m$ from $T_{g}/T^{*}$ Fit')

ax.legend()
ax.grid()

ax.set_xlabel(r'$log(dmax^2)$ fit [mm]')
ax.set_ylabel(r'$log(dmax^2)$ [mm]')

fig.tight_layout()

fig.savefig('../jobs_plots/johnson_m_fit')

df.to_csv('../jobs_data/predictions.txt')
df.to_csv('../jobs_data/johnson_predictions.txt')

print(dfjohnson.columns)
print(df.columns)

pl.show()
