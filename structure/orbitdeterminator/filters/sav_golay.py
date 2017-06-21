'''
Created by Alexandros Kazantzidis
Date 02/06/17

Savintzky Golay: Takes a positional data set (time, x, y, z) and applies the Savintzky Golay filter on it based on the 
polynomial and window parameters we input
'''

from math import *
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
pd.set_option('display.width', 1000)
import pylab
from scipy.signal import savgol_filter
import read_data


def golay(data, window, parameter):
    '''Apply the Savintzky - Golay filter to a positional data set with a 6th order polynomial

        Args:
            data(csv file) = A file containing all of the positional data in the format of (Time, x, y, z)
            window = number for the window of the Savintzky - Golay filter
                     its better to select it as the len(data)/3 and it needs to be an odd number
            parameter = polynomial parameter to be used in the filter

        Return:
            new_positions(numpy array) = filtered data in the same format
    '''

    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    x_new = savgol_filter(x, window, parameter)
    y_new = savgol_filter(y, window, parameter)
    z_new = savgol_filter(z, window, parameter)

    new_positions = np.zeros((len(data), 4))
    new_positions[:, 1] = x_new
    new_positions[:, 2] = y_new
    new_positions[:, 3] = z_new
    new_positions[:, 0] = data[:, 0]

    return new_positions


# if __name__ == "__main__":
#     my_data = read_data.load_data('orbit.csv')
#     window = 21
#     positions_filtered = golay(my_data, window, 6)
#     print(positions_filtered - my_data)
