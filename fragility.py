from matplotlib import pyplot as pl

from functions import finder, data_parse
from math import e

from scipy.constants import physical_constants
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import pandas as pd
import scipy as sc
import numpy as np
import argparse
import ast
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
                    help='The name of the thermodynamic data files.'
                    )

parser.add_argument(
                    '-v',
                    action='store',
                    type=str,
                    help='The name of the viscosity data files.'
                    )

parser.add_argument(
                    '-t',
                    action='store',
                    type=str,
                    help='The file containing the hold temperature.'
                    )

parser.add_argument(
                    '-g',
                    action='store',
                    type=str,
                    help='The file containing Tg.'
                    )

parser.add_argument(
                    '-c',
                    action='store',
                    type=str,
                    help='The viscosity choice for T*.'
                    )

parser.add_argument(
                    '-a',
                    action='store',
                    type=str,
                    help='Location to save analysis data.'
                    )

parser.add_argument(
                    '-p',
                    action='store',
                    type=str,
                    help='Location to save plots.'
                    )


args = parser.parse_args()


def find_nearest(array, value):
    '''
    Find the nearest point.

    inputs:
        array = The data array.
        value = The point in question.

    outputs:
        idx = The index of the nearest value.
    '''

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx


def fragility_plots(
                    fig,
                    ax,
                    x,
                    y,
                    xfit,
                    yfit,
                    visc_cut,
                    label,
                    ):
    '''
    The plotting code for Tg.

    inputs:
        fig = The figure object.
        ax = The axes object.
        x = The temperature data.
        y = The viscosity data.
        xfit = x data used for interpolation.
        yfit = The interpolation between data points.
        visc_cut = The cutoff viscosity.
        label = The label for the data.

    outputs:
        NA
    '''

    line = ax.plot(
                   x,
                   y,
                   marker='.',
                   linestyle='none',
                   )

    color = line[0].get_color()

    ax.plot(
            xfit,
            yfit,
            color=color,
            label=label
            )

    ax.grid()
    ax.legend()

    ax.set_xlabel(r'$1000/Temperature$ $[K^{-1}]$')
    ax.set_ylabel(r'Viscosity $[Pa \cdot s]$')

    fig.tight_layout()


def run_iterator(path, filename, viscname, tgfilename, tempfile, visc_cut):
    '''
    Iterate through each run and gather data.

    inputs:
        path = The path where run data is.
        filename = The name of the file containing thermodynamic data.
        viscname = The name of the file containing viscosity data.
        tgfilename = The name of the file containing Tg data.
        tempfile = The name of the file containing the hold temperature.
        visc_cut = The viscosity choice for T*.

    outputs:
        df = The data from runs.
    '''

    # Gather all applicable paths
    print(filename, path)
    paths = finder(filename, path)

    k = physical_constants['Boltzmann constant in eV/K'][0]

    # Glass transition data
    tg = pd.read_csv(tgfilename)

    # Gather all data and make one dataframe
    df = []
    for item in paths:
        job = os.path.commonprefix([item, path])
        job = item.strip(job)

        cols, data = data_parse(os.path.join(item, filename))
        data = pd.DataFrame(data, columns=cols)

        cols, visc = data_parse(os.path.join(item, viscname))
        visc = pd.DataFrame(visc, columns=cols)

        data = data.merge(visc)

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
    df['atoms'] = df['job'].apply(lambda x: re.split('(\d+)', x))
    df['atoms'] = df['atoms'].apply(lambda x: [i for i in x if i.isnumeric()])
    df['atoms'] = df['atoms'].apply(lambda x: [int(i) for i in x])
    df['atoms'] = df['atoms'].apply(lambda x: sum(x))

    # Group data by run
    groups = df.groupby(['job', 'hold_temp'])
    mean = groups.mean().add_suffix('_mean').reset_index()
    groups = mean.groupby(['job'])

    # Determine Tg from minimum value (only works for super small systems)
    mean = mean.merge(tg, on='job', how='outer')
    groups = mean.groupby(['job'])

    fig, ax = pl.subplots()

    data = {
            'job': [],
            'visc': [],
            'tg': [],
            'tstar': [],
            }

    for i, j in groups:
        data['job'].append(i)

        x = j['temp_mean'].values
        y = j['visc_mean'].values

        tg = np.unique(j['tg'].values)[0]

        # Cut viscosities below Tg
        indexes = (x > tg)
        x = x[indexes]
        y = y[indexes]

        xfit = np.linspace(min(x), max(x), 1000)
        yfit = interp1d(x, y)(xfit)

        idx = find_nearest(yfit, visc_cut)

        tstar = xfit[idx]
        data['visc'].append(yfit[idx])
        data['tg'].append(tg)
        data['tstar'].append(tstar)

        numerator = 1000
        fragility_plots(
                        fig,
                        ax,
                        numerator/x,
                        y,
                        numerator/xfit,
                        yfit,
                        visc_cut,
                        i,
                        )

    ax.axhline(
               visc_cut,
               linestyle=':',
               label='Viscosity Choice',
               color='k',
               )

    pl.close('all')

    data = pd.DataFrame(data)

    return fig, ax, data


fig, ax, data = run_iterator(
                             args.d,
                             args.n,
                             args.v,
                             args.g,
                             args.t,
                             ast.literal_eval(args.c)
                             )

data.to_csv(os.path.join(args.a, 'fragility.txt'), index=False)
fig.savefig(os.path.join(args.p, 'fragility.png'))
print(data)
