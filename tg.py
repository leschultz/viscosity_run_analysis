from matplotlib import pyplot as pl
from functions import finder, data_parse
from scipy.constants import physical_constants
from scipy.interpolate import UnivariateSpline

import pandas as pd
import scipy as sc
import numpy as np
import argparse
import os
import re


parser = argparse.ArgumentParser(
                                 description='Arguments for gathering Tg.'
                                 )

parser.add_argument(
                    '-d',
                    action='store',
                    type=str,
                    help='The location of data.'
                    )

parser.add_argument(
                    '-n',
                    action='store',
                    type=str,
                    help='The name of data files.'
                    )

parser.add_argument(
                    '-t',
                    action='store',
                    type=str,
                    help='The file containing the hold temperature.'
                    )

args = parser.parse_args()


def rmse(act, pred):
    '''
    Calculate the root mean squared error (RMSE).

    inputs:
        act = The actual values.
        pred = The predicted values.
    outputs:
        e = The RMSE.
    '''

    act = np.array(act)
    pred = np.array(pred)

    e = (pred-act)**2
    e = e.mean()
    e = np.sqrt(e)

    return e


def opt(x, y):
    '''
    Linearely fit data from both ends of supplied data and calculate RMSE.

    Left fit:
        Start at a pivot (x[0]) and then fit multiple lines from x[n] until
        x[1]. Calculate RMSE for each fit.

    Right fit:
        Start at a pivot (x[n]) and then fit multiple lines from x[0] until
        x[n-1]. Calculate RMSE for each fit.

    After the fits are completed, their errors are averaged for each
    x point included in a fit excluding the pivots.

    The minimum averaged RMSE is considered to be the optimal point for
    fitting two lines.

    inputs:
        x = Horizontal axis data.
        y = Vertical axis data.
    outputs:
        xcut = The chosen x point.
        endpoints = The non-pivot x values.
        middlermse = The average RMSE.
    '''

    # Make data into numpy arrays
    x = np.array(x)
    y = np.array(y)

    n = len(x)  # The length of data

    ldata = []  # Left fits non-pivot and RMSE
    rdata = []  # Right fits non-pivot and RMSE

    for i in list(range(n-1)):

        # Left fit
        xl = x[:n-i]
        yl = y[:n-i]

        ml, il, _, _, _ = linregress(xl, yl)
        yfitl = ml*xl+il
        rmsel = rmse(yl, yfitl)  # Left RMSE

        # Right fit
        xr = x[i:]
        yr = y[i:]

        mr, ir, _, _, _ = linregress(xr, yr)
        yfitr = mr*xr+ir
        rmser = rmse(yr, yfitr)  # Right RMSE

        # Exclude pivot points from data
        if i > 0:
            ldata.append((xl[-1], rmsel))
            rdata.append((xr[0], rmser))

    # Align data based on pivot point
    ldata = np.array(ldata)
    rdata = np.array(rdata[::-1])

    middle_rmse = (ldata[:, 1]+rdata[:, 1])/2  # Mean RMSE
    mcut = np.argmin(middle_rmse)  # Minimum RMSE index
    xcut = ldata[mcut, 0]  # x data with minimum RMSE

    endpoints = ldata[:, 0]  # Non-pivot points for fits

    return xcut, endpoints, middle_rmse


def tg(df):
    '''
    Calculate the glass transition temperature based on E-3kT.

    inputs:
        df = The data for the run.
    outputs:
    '''
    k = sc.constants.physical_constants['Boltzmann constant in eV/K']

    df = pd.DataFrame()

    e = self.dfsys['TotEng']/self.natoms-3.*k[0]*self.dfsys['Temp']

    df['Temp'] = self.dfsys['Temp']
    df['E-3kT'] = e

    # Use data at and after start of cooling
    condition = self.dfsys['Step'] >= self.hold1
    dfcool = df[condition]
    dfcool = dfcool.sort_values(by=['Temp'])

    x = dfcool['Temp'].values
    y = dfcool['E-3kT'].values

    # Cutoff region
    condition = x <= max_temp
    xcut = x[condition]
    ycut = y[condition]

    # Spline fit of cut region
    k, s = (5, 1)
    spl = UnivariateSpline(x=xcut, y=ycut, k=k, s=s)
    xfitcut = np.linspace(xcut[0], xcut[-1], 100)
    yfitcut = spl(xfitcut)

    tg, endpoints, middle_rmse = opt(xfitcut, yfitcut)


def run_iterator(path, filename, tempfile):
    '''
    Iterate through each run and gather data.

    inputs:
        path = The path where run data is.
        filename = The name of the file containing thermodynamic data.
        tempfile = The name of the file containing the hold temperature.

    outputs:
        df = The data from runs.
    '''

    # Gather all applicable paths
    print(filename, path)
    paths = finder(filename, path)

    k = physical_constants['Boltzmann constant in eV/K'][0]

    # Gather all data and make one dataframe
    df = []
    for item in paths:
        job = os.path.commonprefix([item, path])
        job = item.strip(job)

        cols, data = data_parse(os.path.join(item, filename))
        data = pd.DataFrame(data, columns=cols)

        # Set beggining steps to zero
        data['TimeStep'] -= min(data['TimeStep'])
        data['run'] = job

        # Hold temperature
        temp = float(
                     np.loadtxt(os.path.join(item, tempfile))
                     )  # Hold temperature

        data['hold_temp'] = temp
        df.append(data)

    df = pd.concat(df)
    df['system'] = df['run'].apply(lambda x: x.split('/run/')[0])

    # The number of atoms from naming convention
    df['atoms'] = df['system'].apply(lambda x: re.split('(\d+)', x))
    df['atoms'] = df['atoms'].apply(lambda x: [i for i in x if i.isnumeric()])
    df['atoms'] = df['atoms'].apply(lambda x: [int(i) for i in x])
    df['atoms'] = df['atoms'].apply(lambda x: sum(x))

    # Calculate E-3kT
    df['etot'] = df['pe']+df['ke']
    df['etot/atom'] = df['etot']/df['atoms']
    df['etot/atom-3kT'] = df['etot/atom']-3.0*k*df['hold_temp']

    # Gather specific volume
    df['vol/atom'] = df['vol']/df['atoms']

    df = df.sort_values(by=['temp'])

    groups = df.groupby(['system'])

    for i, j in groups:

        x = j['temp']
        y = j['etot/atom-3kT']
        print(x)

        # Cutoff region
        xcut = x
        ycut = y

        # Spline fit of cut region
        k, s = (5, 1)
        spl = UnivariateSpline(x=xcut, y=ycut, k=k, s=s)
        xfitcut = np.linspace(xcut[0], xcut[-1], 100)
        yfitcut = spl(xfitcut)

        tg, endpoints, middle_rmse = opt(xfitcut, yfitcut)


run_iterator(args.d, args.n, args.t)
