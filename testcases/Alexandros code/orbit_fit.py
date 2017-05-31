'''
Created by Alexandros Kazantzidis
Date 30/05/17
'''

from math import *
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import matplotlib.pylab as plt
import matplotlib as mpl

import lamberts
import orbit_output
import PyKEP as pkp


def create_kep(data):

    '''This function computes all the keplerian elements for every point of the orbit you provide
       It implements a tool for deleting all the points that give extremely jittery state vectors 

        Input
        
        data = read file csv that contains the positional data set in (Time, x, y, z) Format
        
        Output
        
        kep = a numpy array containing all the keplerian elements computed for the orbit given in 
            [semi major axis (a), eccentricity (e), inclination (i), argument of perigee (ω), 
            right ascension of the ascending node (Ω), true anomaly (v)] format 
    '''

    kep1 = np.zeros((len(my_data), 6))
    kep2 = np.zeros((len(my_data), 6))

    v_hold = np.zeros((len(my_data), 3))
    v_abs1 = np.empty([len(my_data)])
    v1, v2 = lamberts.lamberts(my_data[0, :], my_data[1, :])
    v_abs1[0] = (v1[0]**2 + v1[1]**2 + v1[2]**2) ** (0.5)
    v_hold[0] = v1

    ## Produce all the 2 consecutive pairs and find the velocity with lamberts.lamberts() method
    for i in range(1, (len(my_data)-1)):

        j = i + 1
        v1, v2 = lamberts.lamberts(my_data[i, :], my_data[j, :])
        r1 = my_data[i, 1:4]
        r2 = my_data[j, 1:4]

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


    ## collecting the position vector r(x ,y, z) that come along with the velocities with kept above
    final_r = np.zeros((len(store_i), 3))
    j = 0
    for i in store_i:
        final_r[j] = my_data[i, 1:4]
        j += 1

    ## finally we transform the state vectors = position vectors + velocity vectors into keplerian elements
    kep = np.zeros((len(store_i), 6))
    for i in range(0, len(final_r)):

        kep[i] = np.ravel(lamberts.transform(final_r[i], final_v[i]))

    return kep

if __name__ == "__main__":

    my_data = orbit_output.get_data('orbit')
    kep = create_kep(my_data)
    df = pd.DataFrame(kep)
    df = df.rename(columns={0: 'a(km/m)', 1: 'e (number)', 2: 'i (degrees)', 3: 'ω (degrees)',
                            4: 'Ω (degrees)', 5: 'v (degrees)'})
    print("These are the computed keplerian elements for the available points of your orbit")
    print(df)




