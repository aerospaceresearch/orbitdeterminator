'''
Runs the whole process in one file
Input a .csv positional data file (time, x, y, z) and this script generated the final set of keplerian elements
along with a plot and a filtered.csv data file
'''


from util import (read_data, kep_state, rkf78)
from filters import (sav_golay, triple_moving_average, golay_window)
from kep_determination import (lamberts_kalman, interpolation)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
import math
pd.set_option('display.width', 1000)


# First read the csv file called "orbit" with the positional data
data = read_data.load_data("orbit.csv")
# data = np.genfromtxt("orbit.csv")
data = data[1:, :]
data[:, 1:4] = data[:, 1:4] / 1000

# Apply the Triple moving average filter with window = 3
data_after_filter = triple_moving_average.generate_filtered_data(data, 3)

error_apriori = 20
c = golay_window.c(error_apriori)
c = int(c)
if (c % 2) == 0:
    c = c + 1
golay_window = len(data) / c
golay_window = int(golay_window)


# Apply the Savintzky - Golay filter with window = 31 and polynomail parameter = 6
data_after_filter = sav_golay.golay(data_after_filter, golay_window, 3)


# Compute the residuals between filtered data and initial data and then the sum and mean values of each axis
res = data_after_filter[:, 1:4] - data[:, 1:4]
sums = np.sum(res, axis=0)
df_sums = pd.DataFrame(sums)
df_sums = df_sums.rename(index={0: 'x axis', 1: 'y axis',
                                2: 'z axis'})
df_sums = df_sums.rename(columns={0: 'Sum of the residuals (km)'})

means = np.mean(res, axis=0)
df_means = pd.DataFrame(means)
df_means = df_means.rename(index={0: 'x axis', 1: 'y axis',
                                  2: 'z axis'})
df_means = df_means.rename(columns={0: 'Mean of the residuals (km)'})
print(df_sums)
print(" ")
print(df_means)
print(" ")


# Save the filtered data into a new csv called "filtered"
np.savetxt("filtered.csv", data_after_filter, delimiter=",")


# Apply Lambert's solution for the filtered data set
kep_lamb = lamberts_kalman.create_kep(data_after_filter)


# Apply the interpolation method
kep_inter = interpolation.main(data_after_filter)


# Apply Kalman filters to find the best approximation of the keplerian elements for both solutions
# set we a estimate of measurement vatiance R = 0.01 ** 2
kep_final_lamb = lamberts_kalman.kalman(kep_lamb, 0.01 ** 2)
kep_final_lamb = np.transpose(kep_final_lamb)

kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)
kep_final_inter = np.transpose(kep_final_inter)

kep_final = np.zeros((6, 2))
kep_final[:, 0] = np.ravel(kep_final_lamb)
kep_final[:, 1] = np.ravel(kep_final_inter)


# Print the final orbital elements for both solutions
df = pd.DataFrame(kep_final)
df = df.rename(index={0: 'Semi major axis (km)', 1: 'Eccentricity (float number)', 2: 'Inclination (degrees)',
                        3: 'Argument of perigee (degrees)', 4: 'Right ascension of the ascending node (degrees)',
                        5: 'True anomally (degrees)'})
df = df.rename(columns={0: 'Lamberts Results', 1: 'Interpolation Results'})
print(df)


# Plot the initial data set, the filtered data set and the final orbit


# First we transform the set of keplerian elements into a state vector
state = kep_state.kep_state(kep_final_inter)


# Then we produce more state vectors at varius times using a Runge Kutta algorithm
keep_state = np.zeros((6, 150))
ti = 0.0
tf = 1.0
t_hold = np.zeros((150, 1))
x = state
h = 0.1
tetol = 1e-04
for i in range(0, 150):
    keep_state[:, i] = np.ravel(rkf78.rkf78(6, ti, tf, h, tetol, x))
    t_hold[i, 0] = tf
    tf = tf + 1

positions = keep_state[0:3, :]


## Finally we plot the graph
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(data[:, 1], data[:, 2], data[:, 3], ".", label='Initial data ')
ax.plot(data_after_filter[:, 1], data_after_filter[:, 2], data_after_filter[:, 3], "k", linestyle='-',
        label='Filtered data')
ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label='Orbit after Interpolation method')
ax.legend()
ax.can_zoom()
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
plt.show()

