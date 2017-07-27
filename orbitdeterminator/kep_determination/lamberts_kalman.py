'''
Created by Alexandros Kazantzidis
Date : 31/05/17

Lamberts Kalman: Takes a positional data set in the format of (time, x, y, z) and produces one set of six keplerian
elements (a, e, i, ω, Ω, v) using Lambert's solution for preliminary orbit determination and Kalman filters
'''


import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.util import state_kep
import numpy as np
import matplotlib.pylab as plt
import PyKEP as pkp
from math import *
import pandas as pd
pd.set_option('display.width', 1000)


def orbit_trajectory(x1_new, x2_new, time):
    '''Check if we want to keep the result of lamberts() for retrograde or counter-clock wise motion
    
    Args:
        x1(numpy array): time and position for point 1 [time1,x1,y1,z1]
        x2(numpy array): time and position for point 2 [time2,x2,y2,z2]
        time: time difference between the 2 points
        
    Returns:
    	traj(boolean) : True if we want to keep retrogade, False if we want counter-clock wise
    '''

    l = pkp.lambert_problem(x1_new, x2_new, time, 398600.4405, False)

    v1 = l.get_v1()
    v1 = np.asarray(v1)
    v1 = np.reshape(v1, 3)
    x1_new = np.asarray(x1_new)

    kep1 = state_kep.state_kep(x1_new, v1)

    if kep1[0] < 0.0:
        traj = True
    elif kep1[1] > 1.0:
        traj = True
    else:
        traj = False

    return traj


def lamberts(x1, x2, traj):
    '''Takes two position points - numpy arrays with time,x,y,z as elements
       and produces two vectors with the state vector for both positions using Lamberts solution

    Args:
        x1(numpy array): time and position for point 1 [time1,x1,y1,z1]
        x2(numpy array): time and position for point 2 [time2,x2,y2,z2]

    Returns:
        v1(numpy array): velocity vector for point 1 (v1x, v1y, v1z)
    '''

    x1_new = [1, 1, 1]
    x1_new[:] = x1[1:4]
    x2_new = [1, 1, 1]
    x2_new[:] = x2[1:4]
    time = x2[0] - x1[0]

    # traj = orbit_trajectory(x1_new, x2_new, time)

    l = pkp.lambert_problem(x1_new, x2_new, time, 398600.4405, traj)

    v1 = l.get_v1()
    v1 = np.asarray(v1)
    v1 = np.reshape(v1, 3)

    return v1


def check_keplerian(kep):
    '''Checks all the sets of keplerian elements to see if they have wrong values like eccentricity greater that 1 or
       a negative number for semi major axis

     Args:
        kep(numpy array): all the sets of keplerian elements in [semi major axis (a), eccentricity (e),
                          inclination (i), argument of perigee (ω), right ascension of the ascending node (Ω),
                          true anomaly (v)] format
     
     Returns:
        kep_final(numpy array): the final corrected set of keplerian elements that will be inputed in the kalman filter
    '''

    kep_new = list()
    for i in range(0, len(kep)):

        if kep[i, 3] < 0.0:
            kep[i, 3] = 360 + kep[i, 3]
        elif kep[i, 4] < 0.0:
            kep[i, 4] = 360 + kep[i, 4]

        if kep[i, 1] > 1.0:
            pass
        elif kep[i, 0] < 0.0:
            pass
        else:
            kep_new.append(kep[i, :])

    kep_final = np.asarray(kep_new)

    return kep_final


def create_kep(my_data):
    '''Computes all the keplerian elements for every point of the orbit you provide using Lambert's solution
       It implements a tool for deleting all the points that give extremely jittery state vectors

        Args:
            data(numpy array) : contains the positional data set in (Time, x, y, z) Format


        Returns:
            kep(numpy array) : a numpy array containing all the keplerian elements computed for the orbit given in
                               [semi major axis (a), eccentricity (e), inclination (i), argument of perigee (ω),
                               right ascension of the ascending node (Ω), true anomaly (v)] format
    '''
    v_hold = np.zeros((len(my_data), 3))
    # v_abs1 = np.empty([len(my_data)])

    x1_new = [1, 1, 1]
    x1_new[:] = my_data[0, 1:4]
    x2_new = [1, 1, 1]
    x2_new[:] = my_data[1, 1:4]
    time = my_data[1, 0] - my_data[0, 0]
    traj = orbit_trajectory(x1_new, x2_new, time)

    v1 = lamberts(my_data[0, :], my_data[1, :], traj)
    # v_abs1[0] = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** (0.5)
    v_hold[0] = v1

    # Produce all the 2 consecutive pairs and find the velocity with lamberts() method
    for i in range(1, (len(my_data) - 1)):

        j = i + 1
        v1 = lamberts(my_data[i, :], my_data[j, :], traj)

        v_hold[i] = v1
        # compute the absolute value of the velocity vector for every point
        # v_abs1[i] = (v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) ** (0.5)

        # If the value of v_abs(i) > v_abs(0) * 10, then we dont keep that value v(i) because it is propably a bad jiitery product
        # if v_abs1[i] > (10 * v_abs1[0]):
        #     v_hold[i] = v1
        # else:
        #     v_hold[i] = v1

    # we know have lots of [0, 0, 0] inside our numpy array v(vx, vy, vz) and we dont want them because they produce a bug
    # when we'll try to transform these products to keplerian elements
    bo = list()
    store_i = list()
    for i in range(0, len(v_hold)):
        bo.append(np.all(v_hold[i, :] == 0.0))

    for i in range(0, len(v_hold)):
        if bo[i] == False:
            store_i.append(i)

    # keeping only the rows with values and throwing all the [0, 0, 0] arrays
    final_v = np.zeros((len(store_i), 3))
    j = 0
    for i in store_i:
        final_v[j] = (v_hold[i])
        j += 1

    # collecting the position vector r(x ,y, z) that come along with the velocities kept above
    final_r = np.zeros((len(store_i), 3))
    j = 0
    for i in store_i:
        final_r[j] = my_data[i, 1:4]
        j += 1

    # finally we transform the state vectors = position vectors + velocity vectors into keplerian elements
    kep = np.zeros((len(store_i), 6))
    for i in range(0, len(final_r)):
        kep[i] = np.ravel(state_kep.state_kep(final_r[i], final_v[i]))

    kep = check_keplerian(kep)
    # np.savetxt("kep11.csv", kep, delimiter=",")
    return kep


# find the mean value of all keplerian elements set and then do a kalman filtering to find the best fit

def kalman(kep, R):
    '''
    Takes as an input lots of sets of keplerian elements and produces
    the fitted value of them by applying kalman filters

    Args:
        kep(numpy array): containing keplerian elements in this format (a, e, i, ω, Ω, v)
        R : estimate of measurement variance

    Returns:
        final_kep(numpy array): one final set of keplerian elements describing the orbit based on kalman filtering
    '''
    
    # first find the mean values for every keplerian element
    mean_kep = np.zeros((1, 6))
    for i in range(0, 6):
        mean_kep[0, i] = np.mean(kep[:, i])

    # the mean value will be selected as the initial guess

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
