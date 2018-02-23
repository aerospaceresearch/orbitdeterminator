"""
Reads the positional data set from a .csv file
"""

import os
import pickle
import numpy as np

_SOURCE = "../raw data"
_DESTINATION = "../filtered data"


def load_data(filename):
    """ Load data into numpy array.

    Loads the data in numpy array for further processing in tab delimiter format.

    Args:
        filename (string): name of the csv file to be parsed

    Returns:
        numpy array: array of the orbit positions, each point of the orbit is of the
        format (time, x, y, z)
    """

    data = np.genfromtxt(filename, delimiter='\t')
    if np.isnan(data[0]).all():
        data = data[1:]

    return data


def save_orbits(source, destination):
    """ Saves objects returned from load_data.

    Args:
        source: path to raw csv files.
        destination: path where objects need to be saved.
    """
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
