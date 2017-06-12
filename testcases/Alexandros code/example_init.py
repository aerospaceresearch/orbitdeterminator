'''
Created by Alexandros Kazantzidis
Date 03/06/17
'''


from math import *
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import matplotlib.pylab as plt
import matplotlib as mpl
from numpy import genfromtxt

import lamberts
import orbit_output
import orbit_fit
import kep_state
import rkf78
import golay_filter
import read_data
import tripple_moving_average


my_data = read_data.load_data('orbit.csv')
my_data = tripple_moving_average.generate_filtered_data(my_data, 3)
window = 21
my_data = golay_filter.golay(my_data, window)

kep = orbit_fit.create_kep(my_data)
kep_final = orbit_fit.kalman(kep)
kep_final = np.transpose(kep_final)
kep_final2 = np.array([[15711.578566], [0.377617], [90.0], [0.887383], [0.0], [28.357744]])


df2 = pd.DataFrame(kep_final)
df2 = df2.rename(index={0: 'Semi major axis (km)', 1: 'Eccentricity (float number)', 2: 'Inclination (degrees)',
                            3: 'Argument of perigee (degrees)', 4: 'Right ascension of the ascending node (degrees)',
                            5: 'True anomally (degrees)'})
df2 = df2.rename(columns={0: 'Final Results'})
print(df2)


state = kep_state.Kep_state(kep_final)
state2 = kep_state.Kep_state(kep_final2)


keep_state2 = np.zeros((6, 20))
keep_state = np.zeros((6, 20))
ti = 0.0
tf = 100.0
x = state
x2 = state2
h = 1.0
tetol = 1e-04
for i in range(0, 20):

    keep_state[:, i] = np.ravel(rkf78.rkf78(6, ti, tf, h, tetol, x))
    keep_state2[:, i] = np.ravel(rkf78.rkf78(6, ti, tf, h, tetol, x2))

    tf = tf + 100


positions = keep_state[0:3, :]
positions2 = keep_state2[0:3, :]
df = pd.DataFrame(positions)


mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(my_data[:, 1], my_data[:, 2], my_data[:, 3], "o", label='Filtered data with golay')
ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label='Orbit after lamberts - kalman')
ax.plot(positions2[0, :], positions2[1, :], positions2[2, :], "o-", label='Perfect orbit')
ax.legend()
ax.can_zoom()
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
plt.show()
