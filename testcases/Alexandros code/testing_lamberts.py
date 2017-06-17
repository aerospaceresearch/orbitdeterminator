'''
Created by Alexandros Kazantzidis
Date 16/06/17
'''

from math import *
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import matplotlib.pylab as plt
import matplotlib as mpl
from numpy import genfromtxt

import lamberts
import orbit_fit
import kep_state
import golay_filter
import read_data
import tripple_moving_average


## A quick script to test the lamberts - kalman solution
## Not to be included in the main program


my_data = read_data.load_data('orbit.csv')
correct = np.array([[15300], [0.372549], [90.0], [0.854792], [0.0], [28.207374]])
my_data = tripple_moving_average.generate_filtered_data(my_data, 3)
keep = np.zeros((6, 100))
for i in range(9, 100, 2):
    my_data = golay_filter.golay(my_data, i)
    kep = orbit_fit.create_kep(my_data)
    estimate = orbit_fit.kalman(kep)
    estimate = np.transpose(estimate)
    keep[:, i] = np.ravel(correct - estimate)


keep = np.transpose(keep)
print(pd.DataFrame(keep))
