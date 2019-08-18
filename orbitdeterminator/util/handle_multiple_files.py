'''
Handles multiple files argument given as a list of string. Returns
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import warnings
import numpy as np
from util import (read_data, get_format, convert_format)


def handle_multiple_files(file_list):
    '''
    Combine data from multiple files (can be in different observational format) into a single data array

    Args:
        file_list (String): A list of string with each string representing a single data file

    Returns:
        Combined data (Numpy Array): Array of combined data from all files in cartesian format
    '''
    source_path = "example_data/Source"
    data = np.empty((0, 4))
    for file in file_list:
        try:
            file_format = get_format.get_format(source_path + file[-3:].upper() + '/{}'.format(file))
        except:
            warnings.warn("Skiping {} file (Not a valid file type)".format(file))
            continue

        if file_format in ['IOD', 'RDE', 'UK']:
            data = np.append(data, convert_format.convert_format(file, file_format), axis=0)
        elif file_format is 'Cartesian':
            data = np.append(data, read_data.load_data(source_path + file[-3:].upper() + "/{}".format(file)), axis=0)
        else:
            print("Skiping {} file (Observation format is undefined)".format(file))
    return data

if __name__ == "__main__":
    
    data = handle_multiple_files(["orbit_split1.csv", "orbit_uk_split2.txt"])
    print("Files processed: orbit.csv and orbit_uk.txt")
    print("Task completed")