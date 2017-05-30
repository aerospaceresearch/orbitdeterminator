''' 
Created by Alexandros Kazantzidis
Date : 29/05/17
'''

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
import orbit_output
import PyKEP as pkp
from math import *

my_data = orbit_output.get_data('orbit')


def lamberts(x1, x2):
    '''Takes two position points - numpy arrays with time,x,y,z as elements
    and produces two vectors with the state vector for both positions using Lamberts solution

    Input

    x1 = [time1,x1,y1,z1]
    x2 = [time2,x2,y2,z2]
    

    Output

    v1 = velocity vector for position 1 (v1x, v1y, v1z)
    v2 = velocity for position 2 (v2x, v2y, v2z)
    '''

    time = np.array([x2[0] - x1[0]])

    x1_new = [1,1,1]
    x1_new[:] = x1[1:4]
    x2_new = [1, 1, 1]
    x2_new[:] = x2[1:4]
    time = x2[0] - x1[0]
    l = pkp.lambert_problem(x1_new, x2_new, time, 398600.4405, True)

    v1 = l.get_v1()
    v2 = l.get_v2()
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1 = np.reshape(v1, 3)
    v2 = np.reshape(v2, 3)

    return v1, v2


def transform(r, v):
    '''
    This function transforms a state vector to a vector containing the six keplerian elements
    Inputs and outputs in numpy array format
    
    Input
    
    r = position vector [x, y, z]
    v = velocity vector (vx, vy, vz)
    Output
    
    kep = keplerian elements [semi major axis (a), eccentricity (e), inclination (i), argument of perigee (ω), 
          right ascension of the ascending node (Ω), true anomaly (v)]
    '''

    import state_kep

    kep = state_kep.state_kep(r, v)
    return kep


if __name__ == "__main__":

    r1 = my_data[0, 1:4]
    r2 = my_data[1, 1:4]
    v1, v2 = lamberts(my_data[0, :], my_data[1, :])
    kep1 = transform(r1, v1)
    print('These are the velocities for the two first points of your data set')
    print(v1, v2)


