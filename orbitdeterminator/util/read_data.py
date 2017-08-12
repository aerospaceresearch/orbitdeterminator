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
    Loads the data in numpy array for further processing.

    Args:
        filename (string): name of the csv file to be parsed

    Returns:
        numpy array: array of the orbit positions, each point of the orbit is of the
        format (time, x, y, z)
    '''
    orbit_file = list(csv.reader(open(filename, "r"), delimiter = "\t"))

    orbit = []
    for point in orbit_file:
        point_tuple = np.array(point, dtype = np.float)
        orbit.append(point_tuple)

    orbit = np.array(orbit) 
    return orbit

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

if __name__ == "__main__":

    save_orbits(_SOURCE, _DESTINATION)