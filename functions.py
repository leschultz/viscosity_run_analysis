'''
Store generic fuctions.
'''

from ast import literal_eval

import numpy as np
import os


def create_dir(path):
    '''
    Create a directory if it does not already exist

    inputs:
        path = The directory to create.
    '''

    if not os.path.exists(path):
        os.makedirs(path)


def finder(name, source):
    '''
    Find the diretories with a file.

    inputs:
        name = The generic name for files to get path of.
        source = The parent directory of all files.

    outputs:
        paths = The matching paths.
    '''

    # Count all mathching paths
    paths = []
    for item in os.walk(source):

        if name not in item[2]:
            continue

        paths.append(item[0])

    return paths


def input_parse(infile):
    '''
    Parse the input file for important parameters

    inputs:
        infile = The name and path of the input file.

    outputs:
        param = Dictionary containing run paramters.
    '''

    with open(infile) as f:
        for line in f:

            line = line.strip().split(' ')

            if 'units' in line:
                units = line[-1]

            if ('vischold' in line) & ('equal' in line):
                hold = int(line[-1])

            if 'pair_coeff' in line:
                line = [i for i in line if i != '']
                elements = line[4:]

            if 'mytimestep' in line:
                timestep = float(line[-1])

            if ('corlength' in line) & ('equal' in line):
                corlength = int(line[-1])

            if ('sampleinterval' in line) & ('equal' in line):
                sampleinterval = int(line[-1])

    param = {
             'units': units,
             'hold': hold,
             'elements': elements,
             'timestep': timestep,
             'corlength': corlength,
             'sampleinterval': sampleinterval,
             }

    return param


def data_parse(sysfile):
    '''
    Parse the thermodynamic data file

    inputs:
        sysfile = The name and path of the thermodynamic data file.

    outputs:
        columns = The columns for the data.
        data = The data from the file.
    '''

    data = []
    with open(sysfile) as f:
        line = next(f)
        for line in f:

            if '#' in line:
                values = line.strip().split(' ')
                columns = values[1:]
                columns = list(map(lambda x: x.split('_')[-1], columns))

            else:
                values = line.strip().split(' ')
                values = list(map(literal_eval, values))
                data.append(values)

    return columns, data


def lammps_units(style):
    '''
    Define the units by the style used in LAMMPS.

    inputs:
        style = The units style used in LAMMMPS.

    outputs:
        x = A dictionary containing the units.
    '''

    if style == 'metal':
        x = {
             'Volume': r'$\AA^{3}$',
             'Time': r'$ps$',
             'PE': r'$eV$',
             'KE': r'$eV$',
             'Temperature': r'$K$',
             'Pressure': r'$bars$',
             'Viscosity': r'$Pa \cdot s$',
             }

    return x
