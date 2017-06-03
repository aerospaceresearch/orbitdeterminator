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


def golay(data):


    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    x_new = savgol_filter(x, 47, 3)
    y_new = savgol_filter(y, 47, 3)
    z_new = savgol_filter(z, 47, 3)

    new_positions = np.zeros((len(data), 4))
    new_positions[:, 1] = x_new
    new_positions[:, 2] = y_new
    new_positions[:, 3] = z_new
    new_positions[:, 0] = data[:, 0]

    return new_positions

    # mpl.rcParams['legend.fontsize'] = 10
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # ax.plot(x_new[:], y_new[:], z_new[:], "r-", label='Orbit_after_filter visualization')
    # ax.plot(my_data[:, 1], my_data[:, 2], my_data[:, 3], "o-", label='Orbit_before_filter visualization')
    # ax.legend()
    # ax.can_zoom()
    # ax.set_xlabel('x (km)')
    # ax.set_ylabel('y (km)')
    # ax.set_zlabel('z (km)')
    # plt.show()



if __name__ == "__main__":
    my_data = orbit_output.get_data('orbit')
    positions_filtered = golay(my_data)
    print(positions_filtered - my_data)
