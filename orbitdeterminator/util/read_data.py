'''
Reads the positional data set from a .csv file
'''

import os
import csv
import pickle
import numpy as np
import json
import orbitdeterminator.kep_determination.positional_observation_reporting as por

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


def detect_json(filename):
    # detect json
    try:
        json.loads(filename)
        file = {"file" : "json"}
    except:
        file = {"file" : None}

    return file


def detect_iod(filename):
    if por.check_iod_format(filename) == True:
        file = {"file" : "iod"}
    else:
        file = {"file": None}

    return file


def detect_csv(filename):

    file = {"file" : "csv"}

    return file


def detect_file_format(filename):

    file = {"file" : None}

    if os.path.exists(filename):

        # loading in all files and check for all 3 formats we currently support
        file_json = detect_json(filename)
        file_iod = detect_iod(filename)
        file_csv = detect_csv(filename)

        if file_json["file"] == "json":
            return file_json

        elif file_iod["file"] == "iod":
            return file_iod

        elif file_csv["file"] == "csv":
            return file_csv

        else:
            return file


    else:
        # no file available
        return file

if __name__ == "__main__":

    #save_orbits(_SOURCE, _DESTINATION)
    print("detecting file", detect_file_format("../orbit.csv"))
    print("detecting file", detect_file_format("../example_data/SATOBS-ML-19200716.txt"))
    print("detecting file", detect_file_format("../orbit1.csv"))