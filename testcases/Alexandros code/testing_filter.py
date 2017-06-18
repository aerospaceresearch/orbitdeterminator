import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import read_data
from scipy import interpolate
from scipy import signal



def gauss(data):

    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    time = data[:, 0]

    # first, make a function to linearly interpolate the data
    f1 = interpolate.interp1d(time, x)
    f2 = interpolate.interp1d(time, y)
    f3 = interpolate.interp1d(time, z)

    time2 = np.linspace(time[0], time[-1], 1000)

    # compute the function on this finer interval
    xx = f1(time2)
    yy = f2(time2)
    zz = f3(time2)

    # make a gaussian window

    window = signal.gaussian(200, 40)

    # convolve the arrays
    smoothed1 = signal.convolve(xx, window/window.sum(), mode='same')
    smoothed2 = signal.convolve(yy, window/window.sum(), mode='same')
    smoothed3 = signal.convolve(zz, window/window.sum(), mode='same')

    final_data = np.zeros((1000, 4))
    final_data[:, 1] = smoothed1
    final_data[:, 2] = smoothed2
    final_data[:, 3] = smoothed3
    final_data[:, 0] = time2
    return final_data

if __name__ == "__main__":
    my_data = read_data.load_data("orbit.csv")
    final = gauss(my_data)

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(my_data[:, 1], my_data[:, 2], my_data[:, 3], "o", label='Filtered data with golay')
    ax.plot(final[:, 1], final[:, 2], final[:, 3], "o", label='Filtered data with golay')
    # ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label='Orbit after lamberts - kalman')
    # ax.plot(positions2[0, :], positions2[1, :], positions2[2, :], "o-", label='Perfect orbit')
    ax.legend()
    ax.can_zoom()
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')
    plt.show()





