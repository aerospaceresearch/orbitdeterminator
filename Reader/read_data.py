'''
Author : Nilesh Chaturvedi
Date Created : 21/05/2017
'''

import os
import sys
import csv
import pickle
import numpy as np

_SOURCE = "../raw data"
_DESTINATION = "../filtered data"

def load_data(filename):
    ''' Loads the data in numpy array for further processing.
    	
    Args:
        filename: name of the csv file to be parsed

    Returns:
        orbit: 2D numpy array of the orbit.
        Each point of the orbit is of the 
        format : [timestamp, x-coordinate, y-coordinate, z-coordinate]
    '''
    orbit_file = list(csv.reader(open(filename, "r"), delimiter = ","))

    orbit = []
    for point in orbit_file:
        point_tuple = np.array(point, dtype = np.float)
        orbit.append(point_tuple)

    orbit = np.array(orbit)
    return orbit

def save_orbits(source, destination):
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

	# #Returns a dictionary of the format {filename: orbit}
 #    parsed_orbits = {}
 #    files = os.listdir(os.getcwd() + '/' + sys.argv[1])

    