from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from matplotlib import colors as mcolors
from matplotlib import pyplot as pl
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np


def func(x, a, c, d):
    return a*np.exp(-c*x)+d

# Colors for plots
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = [j for i, j in colors.items()]
colors = iter(colors)

dfjohnsonall = pd.read_csv('../../dmax_johnson/johnson/data_clean.csv')
dfjohnsonall = dfjohnsonall[['Alloy', 'Tg (K)', 'TL (K)', 'm', 'dexp (mm)']]
dfjohnsonall.columns = ['composition', 'tg', 'tl', 'm', 'dmax']
dfjohnsonall = dfjohnsonall.dropna()

X_test = dfjohnsonall[['tg', 'tl', 'm']].values

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
df = df[['job', 'Tg (K)', 'TL (K)', 'm', 'dexp (mm)', 'tstar']]
df.columns = ['job', 'tg', 'tl', 'm', 'dmax', 'tstar']

X_train = df[['tg', 'tl', 'm']].values
y_train = df['tstar'].values

X_test = dfjohnsonall[['tg', 'tl', 'm']].values

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)

lin2 = linear_model.LinearRegression() 
lin2.fit(X_poly, y_train) 

y_pred = lin2.predict(poly_reg.fit_transform(X_test))

fig, ax = pl.subplots()

ax.plot(
        X_train[:, 2],
        y_train,
        marker='8',
        linestyle='none',
        label='train'
        )

ax.plot(
        X_test[:, 2],
        y_pred,
        marker='.',
        linestyle='none',
        label='test'
        )

ax.legend()

pl.show()
