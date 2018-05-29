'''
Reads the positional data set from a .csv file
'''

import os
import csv
import pickle
import numpy as np

_SOURCE = "../raw data"
_DESTINATION = "../filtered data"

def load_data(filename):
    '''
    Loads the data in numpy array for further processing in tab delimiter format

    Args:
        filename (string): name of the csv file to be parsed

    Returns:
        numpy array: array of the orbit positions, each point of the orbit is of the
        format (time, x, y, z)
    '''
    return np.genfromtxt(filename, delimiter='\t')[1:]

def save_orbits(source, destination):
    '''
    Saves objects returned from load_data

    Args:
        source: path to raw csv files.
        destination: path where objects need to be saved.
    '''
    if os.path.isdir(source):
        pass
    else:
        os.system("mkdir {}".format(destination))

    for file in os.listdir(source):
        if file.endswith('.csv'):
            orbit = load_data(source + '/' + str(file))
            pickle.dump(orbit, open(destination + "/%s.p" %file[:-4], "wb"))

def load_data_iod_af2(fname):
    '''
    Loads satellite position observation data from IOD-formatted files.
    Currently, the only supported angle format is 2, as specified in IOD format.
    TODO: add other angle formats.

    Args:
        fname (string): name of the IOD-formatted text file to be parsed

    Returns:
        x (numpy array): array of satellite position observations following the
        IOD format, with angle format code = 2.
    '''
    # dt is the dtype for IOD-formatted text files
    dt = 'S15,i8,S1,i8,i8,i8,i8,i8,i8,S1,i8,i8,S1'
    # iod_names correspond to the dtype names of each field
    iod_names = ['object','station','stationstatus','utcdate','utctime','timeunc','angformat','epoch','ra','decsgn','dec','posunc','optical','vismag']
    # TODO: read first line, get sub-format, construct iod_delims from there
    # as it is now, it only works for the given test file iod_data.txt
    # iod_delims are the fixed-width column delimiter followinf IOD format description
    iod_delims = [15,5,2,9,9,3,2,1,8,1,6,3,2,2]
    return np.genfromtxt(fname, dtype=dt, names=iod_names, delimiter=iod_delims, autostrip=True)

if __name__ == "__main__":

    save_orbits(_SOURCE, _DESTINATION)