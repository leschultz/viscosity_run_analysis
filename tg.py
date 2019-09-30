from matplotlib import pyplot as pl

from scipy.constants import physical_constants
from scipy.interpolate import UnivariateSpline
from functions import finder, data_parse
from scipy.stats import linregress
from knee import opt

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

parser.add_argument(
                    '-k',
                    action='store',
                    type=str,
                    help='Spline fitting paramter for degree of smoothing spline.'
                    )

parser.add_argument(
                    '-s',
                    action='store',
                    type=str,
                    help='Spline fitting parameter for smoothing factor.'
                    )


parser.add_argument(
                    '-a',
                    action='store',
                    type=str,
                    help='The location to save data.'
                    )


args = parser.parse_args()


def tg_plots(
             x,
             y,
             k,
             s,
             ):
    '''
    The plotting code for Tg.

    inputs:
        x = The temperature data.
        y = The E-3kT data.
        k = Spline fitting parameter.
        s = Spline fitting parameter.

    outputs:
        tg = The glass transition temperature.
    '''

    minx = x[x.argsort()[:6][-1]]  # Minimum number of points for spline to work
    maxx = max(x)

    rangex = np.linspace(minx, maxx, 100)

    tgs = []
    for i in rangex:
        index = (x <= i)
        xcut = x[index]
        ycut = y[index]

        spl = UnivariateSpline(x=xcut, y=ycut, k=k, s=s)
        xfitcut = np.linspace(min(xcut), max(xcut), 100)
        yfitcut = spl(xfitcut)

        tg, endpoints, middle_rmse = opt(xfitcut, yfitcut)

        tgs.append(tg)

    tgs = np.array(tgs)

    index = np.argmax(tgs)
    tg = tgs[index]
    xcut = rangex[index]

    fig, ax = pl.subplots()

    ax.plot(rangex, tgs, marker='.', linestyle='none', label='Data')

    ax.set_xlabel('Upper Temperature Cutoff [K]')
    ax.set_ylabel('Tg [K]')

    ax.grid()

    fig.tight_layout()

    index = (x <= xcut)
    xnew = x[index]
    ynew = y[index]

    spl = UnivariateSpline(x=xnew, y=ynew, k=k, s=s)
    xfitcut = np.linspace(xnew[0], xnew[-1], 100)
    yfitcut = spl(xfitcut)

    tg, endpoints, middle_rmse = opt(xfitcut, yfitcut)

    ax.axvline(
               xcut,
               linestyle=':',
               color='k',
               label='Upper Temperature Cutoff: '+str(xcut)+' [K]'
               )

    ax.legend(loc='best')

    fig, ax = pl.subplots()

    ax.plot(
            xnew,
            ynew,
            marker='.',
            linestyle='none',
            color='b',
            label='data'
            )

    ax.plot(
            xfitcut,
            yfitcut,
            linestyle=':',
            color='k',
            label='Spline Fit: (k='+str(k)+', s='+str(s)+')'
            )

    ax.axvline(
               tg,
               linestyle='--',
               color='g',
               label='Tg = '+str(tg)+' [K]'
               )

    ax.grid()
    ax.legend()

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('E-3kT [K/atom]')

    fig.tight_layout()

    fig, ax = pl.subplots()

    ax.plot(
            x,
            y,
            marker='.',
            linestyle='none',
            color='b',
            label='data'
            )

    ax.axvline(
               tg,
               linestyle='--',
               color='g',
               label='Tg = '+str(tg)+' [K]'
               )

    ax.grid()
    ax.legend()

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('E-3kT [K/atom]')

    fig.tight_layout()

    pl.close('all')

    return tg


def run_iterator(path, filename, tempfile, k, s):
    '''
    Iterate through each run and gather data.

    inputs:
        path = The path where run data is.
        filename = The name of the file containing thermodynamic data.
        tempfile = The name of the file containing the hold temperature.
        k = Spline fitting parameter
        s = Spline fitting parameter

    outputs:
        df = The data from runs.
    '''

    # Gather all applicable paths
    print(filename, path)
    paths = finder(filename, path)

    kboltz = physical_constants['Boltzmann constant in eV/K'][0]

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
    df['job'] = df['run'].apply(lambda x: x.split('/run/')[0])

    # The number of atoms from naming convention
    df['atoms'] = df['job'].apply(lambda x: re.split('(\d+)', x.split('_')[1]))
    df['atoms'] = df['atoms'].apply(lambda x: [i for i in x if i.isnumeric()])
    df['atoms'] = df['atoms'].apply(lambda x: [int(i) for i in x])
    df['atoms'] = df['atoms'].apply(lambda x: sum(x))

    # Calculate E-3kT
    df['etot'] = df['pe']+df['ke']
    df['etot/atoms'] = df['etot']/df['atoms']
    df['etot/atoms-3kT'] = df['etot/atoms']-3.0*kboltz*df['hold_temp']

    # Group data by run
    groups = df.groupby(['job', 'hold_temp'])
    mean = groups.mean().add_suffix('_mean').reset_index()
    groups = mean.groupby(['job'])

    runs = []
    tgs = []
    for i, j in groups:

        print(i)
        x = j['temp_mean'].values
        y = j['etot/atoms-3kT_mean'].values

        tg = tg_plots(
                      x,
                      y,
                      k,
                      s,
                      )

        runs.append(i)
        tgs.append(tg)


    df = pd.DataFrame({'job': runs, 'tg': tgs})

    return df


df = run_iterator(args.d, args.n, args.t, args.k, args.s)

df.to_csv(os.path.join(args.a, 'tg.txt'), index=False)

