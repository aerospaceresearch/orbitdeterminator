'''
Created by Alexandros Kazantzidis
Date : 29/05/17
'''

import numpy as np
import matplotlib.pylab as plt
import PyKEP as pkp
from math import *
import read_data



def lamberts(x1, x2):
    '''
    Takes two position points - numpy arrays with time,x,y,z as elements
    and produces two vectors with the state vector for both positions using Lamberts solution

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
    v2 = l.get_v2()
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1 = np.reshape(v1, 3)
    v2 = np.reshape(v2, 3)


    check = (np.greater(v1, np.array([100.0, 100.0, 100.0])))
    check2 = (np.less(v1, np.array([-100.0, -100.0, -100.0])))

    if np.any(check) or np.any(check2) == True:
        l = pkp.lambert_problem(x1_new, x2_new, time, 398600.4405, True)

    v1 = l.get_v1()
    v2 = l.get_v2()
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1 = np.reshape(v1, 3)
    v2 = np.reshape(v2, 3)

    return v1


def transform(r, v):
    '''
    Transforms a state vector to a vector containing the six keplerian elements
    Inputs and outputs in numpy array format
    
    Args:
        r(numpy arrray) = position vector [x, y, z]
        v(numpy arrray) = velocity vector (vx, vy, vz)

    Returns:
        kep(numpy array) = keplerian elements [semi major axis (a), eccentricity (e), inclination (i), 
                           argument of perigee (ω), right ascension of the ascending node (Ω), true anomaly (v)]
    '''

    import state_kep

    kep = state_kep.state_kep(r, v)
    return kep


if __name__ == "__main__":

    my_data = read_data.load_data('orbit.csv')
    r1 = my_data[0, 1:4]
    r2 = my_data[1, 1:4]
    v1 = lamberts(my_data[0, :], my_data[1, :])
    kep1 = transform(r1, v1)
    print('This the velocity for the first point of your data set')
    print(v1)


