'''
Created by Alexandros Kazantzidis
Date : 31/05/17

Lamberts Kalman: Takes a positional data set in the format of (time, x, y, z) and produces one set of six keplerian
elements (a, e, i, ω, Ω, v) using Lambert's solution for preliminary orbit determination and Kalman filters
'''

import sys
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from util import state_kep
import numpy as np
import matplotlib.pylab as plt
import PyKEP as pkp
from math import *
import read_data
import pandas as pd
pd.set_option('display.width', 1000)


def lamberts(x1, x2):
    '''Takes two points (time,x,y,z) and produces two vectors with the state vector for both positions 
    using Lamberts solution

    Args:
        x1(numpy array) = time and position for point 1 [time1,x1,y1,z1]
        x2(numpy array) = time and position for point 2 [time2,x2,y2,z2]

    Returns:
        v1(numpy array) = velocity vector for point 1 (v1x, v1y, v1z)
        v2(numpy array) = velocity for point 2 (v2x, v2y, v2z)
    '''

    time = np.array([x2[0] - x1[0]])

    x1_new = [1, 1, 1]
    x1_new[:] = x1[1:4]
    x2_new = [1, 1, 1]
    x2_new[:] = x2[1:4]
    time = x2[0] - x1[0]

    l = pkp.lambert_problem(x1_new, x2_new, time, 398600.4405, False)

    v1 = l.get_v1()
    v1 = np.asarray(v1)
    v1 = np.reshape(v1, 3)

    check = (np.greater(v1, np.array([100.0, 100.0, 100.0])))
    check2 = (np.less(v1, np.array([-100.0, -100.0, -100.0])))

    if np.any(check) or np.any(check2) == True:
        l = pkp.lambert_problem(x1_new, x2_new, time, 398600.4405, True)

    v1 = l.get_v1()
    v1 = np.asarray(v1)
    v1 = np.reshape(v1, 3)

    return v1


def create_kep(my_data):
    '''Computes all the keplerian elements for every point of the orbit you provide
       It implements a tool for deleting all the points that give extremely jittery state vectors

        Args:  
            data(csv file) = read file csv that contains the positional data set in (Time, x, y, z) Format


        Returns:
            kep(numpy array) = a numpy array containing all the keplerian elements computed for the orbit given in
                               [semi major axis (a), eccentricity (e), inclination (i), argument of perigee (ω),
                               right ascension of the ascending node (Ω), true anomaly (v)] format
    '''

    v_hold = np.zeros((len(my_data), 3))
    v_abs1 = np.empty([len(my_data)])
    v1 = lamberts(my_data[0, :], my_data[1, :])
    v_abs1[0] = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** (0.5)
    v_hold[0] = v1

    ## Produce all the 2 consecutive pairs and find the velocity with lamberts.lamberts() method
    for i in range(1, (len(my_data) - 1)):

        j = i + 1
        v1 = lamberts(my_data[i, :], my_data[j, :])

        ##compute the absolute value of the velocity vector for every point
        v_abs1[i] = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** (0.5)

        ## If the value of v_abs(i) > v_abs(0) * 10, then we dont keep that value v(i) because it is propably a bad jiitery product
        if v_abs1[i] > (10 * v_abs1[0]):
            pass
        else:
            v_hold[i] = v1

    ## we know have lots of [0, 0, 0] indead our numpy array v(vx, vy, vz) and we dont want them because they produce a bug
    ## when we'll try to transform these products to keplerian elements
    bo = list()
    store_i = list()
    for i in range(0, len(v_hold)):
        bo.append(np.all(v_hold[i, :] == 0.0))

    for i in range(0, len(v_hold)):
        if bo[i] == False:
            store_i.append(i)

    ## keeping only the rows with values and throwing all the [0, 0, 0] arrays
    final_v = np.zeros((len(store_i), 3))
    j = 0
    for i in store_i:
        final_v[j] = (v_hold[i])
        j += 1

    ## collecting the position vector r(x ,y, z) that come along with the velocities kept above
    final_r = np.zeros((len(store_i), 3))
    j = 0
    for i in store_i:
        final_r[j] = my_data[i, 1:4]
        j += 1

    ## finally we transform the state vectors = position vectors + velocity vectors into keplerian elements
    kep = np.zeros((len(store_i), 6))
    for i in range(0, len(final_r)):
        kep[i] = np.ravel(state_kep.state_kep(final_r[i], final_v[i]))

    ## check in every row to see if eccentricity is over 1 then the solution is completely wrong and needs
    ## to be deleted
    kep_new = list()
    for i in range(0, len(kep)):
        if kep[i, 1] > 1.0:
            pass
        else:
            kep_new.append(kep[i, :])

    kep = np.asarray(kep_new)

    return kep


## find the mean value of all keplerian elements set and then do a kalman filtering to find the best fit

def kalman(kep, R):
    '''
    Takes as an input lots of sets of keplerian elements and produces
    the fitted value of them by applying kalman filters

    Args:
        kep(numpy array) = containing keplerian elements in this format
        (a, e, i, ω, Ω, v)
        R = estimate of measurement variance

    Returns:
        final_kep(numpy array) = one final set of keplerian elements describing the orbit based on kalman filtering
    '''

    ## first find the mean values for every keplerian element

    mean_kep = np.zeros((1, 6))
    for i in range(0, 6):
        mean_kep[0, i] = np.mean(kep[:, i])

    ## the mean value will be selected a the initial guess

    x_final = np.zeros((1, 6))
    for i in range(0, 6):

        # intial parameters
        n_iter = len(kep)
        sz = n_iter  # size of array
        z = np.zeros((sz, 6))
        z[:, i] = kep[:, i]

        Q = 1e-8  # process variance

        xhat = np.zeros((sz, 6))  # a posteri estimate of x
        P = np.zeros((sz, 6))  # a posteri error estimate
        xhatminus = np.zeros((sz, 6))  # a priori estimate of x
        Pminus = np.zeros((sz, 6))  # a priori error estimate
        K = np.zeros((sz, 6))  # gain or blending factor



        # intial guesses
        xhat[0, i] = mean_kep[0, i]
        P[0, i] = 1.0

        for k in range(1, n_iter):
            # time update
            xhatminus[k, i] = xhat[k - 1, i]
            Pminus[k, i] = P[k - 1, i] + Q

            # measurement update
            K[k, i] = Pminus[k, i] / (Pminus[k, i] + R)

            xhat[k, i] = xhatminus[k, i] + K[k, i] * (z[k, i] - xhatminus[k, i])
            P[k, i] = (1 - K[k, i]) * Pminus[k, i]

        x_final[:, i] = xhat[k, i]

    return x_final

#
# if __name__ == "__main__":
#
#     my_data = read_data.load_data('orbit.csv')
#     kep = create_kep(my_data)
#     df = pd.DataFrame(kep)
#     df = df.rename(columns={0: 'a(km or m)', 1: 'e (number)', 2: 'i (degrees)', 3: 'ω (degrees)',
#                             4: 'Ω (degrees)', 5: 'v (degrees)'})
#     print("These are the computed keplerian elements for the available points of your orbit")
#     print(df)
#
#     kep_final = kalman(kep, 0.01 ** 2)
#     df1 = pd.DataFrame(kep_final)
#     df1 = df1.rename(columns={0: 'a(km or m)', 1: 'e (number)', 2: 'i (degrees)', 3: 'ω (degrees)',
#                               4: 'Ω (degrees)', 5: 'v (degrees)'})
#
#     user = input('Press ENTER to see kalman filters solution')
#     print(" ")
#     print("These are the final fitted values for the keplerian elements based on kalman filtering")
#     print(df1)
#




