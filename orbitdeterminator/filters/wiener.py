'''
Takes a positional data set (time, x, y, z) and applies the Wiener's filter on it based on the
window parameters we input
'''

from math import *
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scipy.signal import wiener
from util import read_data


def wiener_new(data,window):


	'''
    Apply the Wiener's filter to a positional data set.

    Args:
        data (numpy array): containing all of the positional data in the format of (time, x, y, z)
        window (int): window size of the Wiener filter
        

    Returns:
        numpy array: filtered data in the same format
	'''
	x=data[:,1]
	y=data[:,2]
	z=data[:,3]

	x_new=wiener(x,window)
	y_new=wiener(y,window)
	z_new=wiener(z,window)


	new_positions = np.zeros((len(data), 4))
	new_positions[:, 1] = x_new
	new_positions[:, 2] = y_new
	new_positions[:, 3] = z_new
	new_positions[:, 0] = data[:, 0]

	return new_positions
