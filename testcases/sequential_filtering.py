"""
Author : Nilesh Chaturvedi
Date Created : 12th June, 2017

Analysis of varying parameres with filters in sequence
"""
import os
import mse
import numpy
from scipy.signal import savgol_filter

import read_data as rd
import tripple_moving_average as tma


def golay(data, window, degree):

    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    x_new = savgol_filter(x, window, degree)
    y_new = savgol_filter(y, window, degree)
    z_new = savgol_filter(z, window, degree)

    new_positions = numpy.zeros((len(data), 4))
    new_positions[:, 1] = x_new
    new_positions[:, 2] = y_new
    new_positions[:, 3] = z_new
    new_positions[:, 0] = data[:, 0]

    return new_positions

signal = rd.load_data(os.getcwd() + '/orbit0jittery.csv')
perfect = rd.load_data(os.getcwd() + '/orbit0perfect.csv')

errors = []
for tma_window in range(2, 9):
    tma_filtered = tma.generate_filtered_data(signal, tma_window)
    for golay_degree in range(1, 9):
        for golay_window in range(9, 99, 2):
            golay_filtered = golay(tma_filtered, golay_window, golay_degree)
            error = mse.error(perfect, golay_filtered)
            err_tup = [tma_window, golay_window, golay_degree, error]
            errors.append(err_tup)

errors = numpy.array(errors)
errors = errors[numpy.argsort(errors[:,3])]

print("The best filter combination has the following configuration"
    "\n\nTriple Moving Average Window {} \nGolay Window {} \nGolay Degree {}"
    "\nError {}".format(errors[0][0], errors[0][1], errors[0][2], errors[0][3]))
