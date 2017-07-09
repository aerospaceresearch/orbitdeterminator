'''
Runs the whole process in one file
Input a .csv positional data file (time, x, y, z) and this script generated the final set of keplerian elements
along with a plot and a filtered.csv data file
'''


from util import read_data
from util import kep_state
from util import rkf78
from filters import sav_golay
from filters import triple_moving_average
from kep_determination import lamberts_kalman
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
pd.set_option('display.width', 1000)


## First read the csv file called "orbit" with the positional data
data = read_data.load_data("orbit.csv")


## Apply the Triple moving average filter with window = 3
data_after_filter = triple_moving_average.generate_filtered_data(data, 3)


## Apply the Savintzky - Golay filter with window = 31 and polynomail parameter = 6
data_after_filter = sav_golay.golay(data_after_filter, 61, 6)


## Save the filtered data into a new csv called "filtered"
np.savetxt("filtered.csv", data_after_filter, delimiter=",")


## Apply Lambert's solution for the filtered data set
kep = lamberts_kalman.create_kep(data_after_filter)


## Apply Kalman filters to find the best approximation of the keplerian elements set we a estimate of measurement
## vatiance R = 0.01 ** 2
kep_final = lamberts_kalman.kalman(kep, 0.01 ** 2)
kep_final = np.transpose(kep_final)

## Print the final orbital elements
df = pd.DataFrame(kep_final)
df = df.rename(index={0: 'Semi major axis (km)', 1: 'Eccentricity (float number)', 2: 'Inclination (degrees)',
                        3: 'Argument of perigee (degrees)', 4: 'Right ascension of the ascending node (degrees)',
                        5: 'True anomally (degrees)'})
df = df.rename(columns={0: 'Final Results'})
print(df)


## Plot the initial data set, the filtered data set and the final orbit


## First we transform the set of keplerian elements into a state vector
state = kep_state.kep_state(kep_final)


## Then we produce more state vectors at varius times using a Runge Kutta algorithm
keep_state = np.zeros((6, 20))
ti = 0.0
tf = 100.0
t_hold = np.zeros((20, 1))
x = state
h = 1.0
tetol = 1e-04
for i in range(0, 20):
    keep_state[:, i] = np.ravel(rkf78.rkf78(6, ti, tf, h, tetol, x))
    t_hold[i, 0] = tf
    tf = tf + 100

positions = keep_state[0:3, :]


## Finally we plot the graph
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(data[:, 1], data[:, 2], data[:, 3], ".", label='Initial data ')
ax.plot(data_after_filter[:, 1], data_after_filter[:, 2], data_after_filter[:, 3], "k", linestyle='-',
        label='Filtered data')
ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label='Orbit after Lamberts - Kalman')
ax.legend()
ax.can_zoom()
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
plt.show()

