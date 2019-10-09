from matplotlib import pyplot as pl
from os.path import join
from functions import *

import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
                                 description='Arguments for GK viscosity runs.'
                                 )

parser.add_argument(
                    '-p',
                    action='store',
                    type=str,
                    help='The location of runs.'
                    )

parser.add_argument(
                    '-d',
                    action='store',
                    type=str,
                    help='The location to store data.'
                    )

parser.add_argument(
                    '-w',
                    action='store',
                    type=str,
                    help='The location to store plots.'
                    )

parser.add_argument(
                    '-i',
                    action='store',
                    type=str,
                    help='Name of the input file.'
                    )

parser.add_argument(
                    '-s',
                    action='store',
                    type=str,
                    help='Name of file containing themodynamic data.'
                    )

parser.add_argument(
                    '-t',
                    action='store',
                    type=str,
                    help='Name of file containing trajectory data.'
                    )

parser.add_argument(
                    '-l',
                    action='store',
                    type=str,
                    help='Name of file containing the hold temperature.'
                    )

parser.add_argument(
                    '-v',
                    action='store',
                    type=str,
                    help='Generic name of file containing viscosity data.'
                    )

args = parser.parse_args()


def run_data(path, input_name, system_name, traj_name, temp_name, visc_name):
    '''
    Arrange the data for a single run.

    inputs:
        path = The path for the run.
        input_name = The name of the input file.
        system_name = The name of the file containing thermodynamic data.
        traj_name = The name of the file cotaining trajectory data.
        temp_name = The name of the file contining the hold temperature
        visc_name = The name for the file containg viscosity data.

    outputs:
        df = The data for a run.
        units = The units used for a run.
    '''

    # The name and path of files
    inp = join(path, input_name)
    sys = join(path, system_name)
    traj = join(path, traj_name)
    temp = join(path, temp_name)
    visc = join(path, visc_name)

    params = input_parse(inp)  # Gather input file parameters
    temp = np.loadtxt(temp)  # Hold temperature

    units = lammps_units(params['units'])  # The working units

    # Thermodynamic Data
    cols_thermo, df_thermo = data_parse(sys)
    df_thermo = pd.DataFrame(df_thermo, columns=cols_thermo)

    # Viscosity Data
    cols_visc, df_visc = data_parse(visc)
    df_visc = pd.DataFrame(df_visc, columns=cols_visc)

    # Combine the data
    df = df_thermo.merge(df_visc)

    # Calculate and shift time so that the minimum time is zero
    df['TimeStep'] -= min(df['TimeStep'])
    df['time'] = df['TimeStep'].apply(lambda x: x*params['timestep'])

    # The hold temperature
    df['hold_temp'] = temp

    return df, units


def run_plot(df, units):
    '''
    Plot the data for a run.

    inputs:
        df = The data for a run.
        units = The units used for a run.

    outputs:
    '''

    time = df['time'].values
    visc = df['visc'].values

    holdtemp = np.unique(df['hold_temp'].values)[0]

    fig, ax = pl.subplots()

    ax.plot(
            time,
            visc,
            marker='.',
            linestyle='none',
            label=str(holdtemp)+' [K]'
            )

    ax.legend()
    ax.grid()

    ax.set_xlabel('Time ['+units['Time']+']')
    ax.set_ylabel('Green-Kubo Viscosity ['+units['Viscosity']+']')

    fig.tight_layout()

    pl.close('all')

    return fig, ax


def run_iterator(
                 path,
                 data_path,
                 plot_path,
                 *args,
                 **kwargs
                 ):
    '''
    Iterate through each run and gather data.

    inputs:
        path = The path where all runs are stored.
        data_path = The path to store data gathered.
        plot_path = The path to store data plots.

    outputs:
        df_all = The viscosity data from runs.
    '''

    paths = finder(args[0], path)  # Runs to evaluate
    runs = str(len(paths)-1)  # The number of runs to evaluate

    df_all = pd.DataFrame(columns=['run', 'temp', 'visc'])

    count = 0
    for item in paths:

        # The name of the run
        job = os.path.commonprefix([item, path])
        job = item.strip(job)

        try:

            # Progress print
            print('Analyzing ('+str(count)+'/'+runs+'): '+job)

            df, units = run_data(item, *args, **kwargs)  # Data
            fig, ax = run_plot(df, units)  # Plots

            # Gather the hold temperature
            hold_temp = np.unique(df['hold_temp'].values)[0]

            # Gather the final viscosity values
            visc_final = df['visc'].values[-1]

            # Collect the job id, hold temperature, and final viscosity
            df_all.loc[count] = [job, hold_temp, visc_final]

            datastore = join(data_path, job)  # Data path
            plotstore = join(plot_path, job)  # Plot path

            # Create data and plot path if they do not exist
            create_dir(datastore)
            create_dir(plotstore)

            # Save Data
            df.to_csv(join(datastore, 'data.txt'), index=False)

            # Save plots
            fig.savefig(join(plotstore, 'gk_viscosity_vs_time'))

            count += 1

        except Exception:
            print('Problem with: '+job)

    return df_all


df = run_iterator(
                  args.p,
                  args.d,
                  args.w,
                  args.i,
                  args.s,
                  args.t,
                  args.l,
                  args.v
                  )

df.to_csv(join(args.d, 'viscosities.txt'), index=False)
