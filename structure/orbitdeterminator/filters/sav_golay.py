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
from scipy.signal import savgol_filter
from util import read_data



def golay(data, window, degree):
    '''Apply the Savintzky-Golay filter to a positional data set.

       Args:
           data (csv file): A file containing, all of the positional data in the format of (time, x, y, z).
           window (int): window size of the Savintzky-Golay filter.
           degree (int): degree of the ploynomial in Savintzky-Golay filter.

        Return:
            new_positions (numpy array): filtered data in the same format
    '''

    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    x_new = savgol_filter(x, window, degree)
    y_new = savgol_filter(y, window, degree)
    z_new = savgol_filter(z, window, degree)


    new_positions = np.zeros((len(data), 4))
    new_positions[:, 1] = x_new
    new_positions[:, 2] = y_new
    new_positions[:, 3] = z_new
    new_positions[:, 0] = data[:, 0]

    return new_positions


#
# if __name__ == "__main__":
#
#     pd.set_option('display.width', 1000)
#     my_data = read_data.load_data('orbit.csv')
#     window = 21 # its better to select it as the len(data)/3 and it needs to be an odd number
#     degree = 6
#     positions_filtered = golay(my_data, window, degree)
#     print(positions_filtered - my_data)
#
