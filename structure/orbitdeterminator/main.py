'''
Runs the whole process in one file
Input a .csv positional data file (time, x, y, z) and this script generated the final set of keplerian elements
'''


import read_data
from filters import sav_golay
from filters import triple_moving_average
from kep_determination import lamberts_kalman

## First read the csv file called "orbit" with the positional data
data = read_data.load_data("orbit.csv")


## Apply the Triple moving average filter with window = 3
data_after_triple = triple_moving_average.generate_filtered_data(data, 3)


## Apply the Savintzky - Golay filter with window = 31 and polynomail parameter = 6
data_after_golay = sav_golay.golay(data_after_triple, 31, 6)


## Apply Lambert's solution for the filtered data set
kep = lamberts_kalman.create_kep(data_after_golay)


## Apply Kalman filters to find the best approximation of the keplerian elements set we a estimate of measurement
## vatiance R = 0.01 ** 2
kep_final = lamberts_kalman.kalman(kep, 0.01 ** 2)


print(kep_final)
