'''
Created by Alexandros Kazantzidis
Date 02/06/17
'''


from math import *
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import matplotlib.pylab as plt
import pylab
from scipy.signal import savgol_filter

import orbit_output


def golay(data, window):


    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    x_new = savgol_filter(x, window, 3)
    y_new = savgol_filter(y, window, 3)
    z_new = savgol_filter(z, window, 3)

    new_positions = np.zeros((len(data), 4))
    new_positions[:, 1] = x_new
    new_positions[:, 2] = y_new
    new_positions[:, 3] = z_new
    new_positions[:, 0] = data[:, 0]

    return new_positions


if __name__ == "__main__":
    my_data = orbit_output.get_data('orbit')
    window = 41
    positions_filtered = golay(my_data, window)
    print(positions_filtered - my_data)
