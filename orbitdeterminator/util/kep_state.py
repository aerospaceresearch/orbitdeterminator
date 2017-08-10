'''
Takes a set of keplerian elements (a, e, i, ω, Ω, v) and transforms it into a state vector
(x, y, z, vx, vy, vz) where v is the velocity of the satellite
'''

import numpy as np
from math import *


def kep_state(kep):
    '''
    Converts the keplerian elements to position and velocity vector

    Args:
        kep(numpy array): a 1x6 matrix which contains the following variables
            kep(0): semi major axis (km)
            kep(1): eccentricity (number)
            kep(2): inclination (degrees)
            kep(3): argument of perigee (degrees)
            kep(4): right ascension of the ascending node (degrees)
            kep(5): true anomaly (degrees)

    Returns:
        numpy array: 1x6 matrix which contains the position and velocity vector
        r(0),r(1),r(2): position vector (x,y,z) km
        r(3),r(4),r(5): velocity vector (vx,vy,vz) km/s
    '''

    r = np.zeros((6, 1))
    mu = 398600.4405

    # unload orbital elements array

    sma = kep[0, 0]
    ecc = kep[1, 0]
    inc = kep[2, 0]
    inc = radians(inc)
    argper = kep[3, 0]
    argper = radians(argper)
    raan = kep[4, 0]
    raan = radians(raan)
    tanom = kep[5, 0]
    tanom = radians(tanom)

    slr = sma * (1 - ecc * ecc)
    rm = slr / (1 + ecc * cos(tanom))

    arglat = argper + tanom  # argument of latitude

    sarglat = sin(arglat)
    carglat = cos(arglat)

    c4 = sqrt(mu / slr)
    c5 = ecc * cos(argper) + carglat
    c6 = ecc * sin(argper) + sarglat

    sinc = sin(inc)
    cinc = cos(inc)

    sraan = sin(raan)
    craan = cos(raan)

    # position vector
    r[0, 0] = rm * (craan * carglat - sraan * cinc * sarglat)
    r[1, 0] = rm * (sraan * carglat + cinc * sarglat * craan)
    r[2, 0] = rm * sinc * sarglat

    # velocity vector

    r[3, 0] = -c4 * (craan * c6 + sraan * cinc * c5)
    r[4, 0] = -c4 * (sraan * c6 - craan * cinc * c5)
    r[5, 0] = c4 * c5 * sinc

    # # transform r and v into ECI frame
	#
    # R1inc = np.array([[1, 0, 0],
    #                   [0, cos(-inc), sin(-inc)],
    #                   [0, -sin(-inc), cos(-inc)]
    #                   ])
    # R3raan = np.array([[cos(-raan), sin(-raan), 0],
    #                   [-sin(-raan), cos(-raan), 0],
    #                   [0, 0, 1]
    #                    ])
    # R3argper = np.array([[cos(-argper), sin(-argper), 0],
    #                     [-sin(-argper), cos(-argper), 0],
    #                     [0, 0, 1]
    #                      ])
	#
    # r_final1 = np.dot(R3raan, R1inc)
    # r_final2 = np.dot(R3argper, r[0:3])
	#
    # r_final = np.dot(r_final1, r_final2)
    # print(r_final)
    # v_final1 = np.dot(R3raan, R1inc)
    # v_final2 = np.dot(R3argper, r[3:6])
    # v_final = np.dot(v_final1, v_final2)
    # print(v_final)
	#
    # r[0:3] = r_final
    # r[3:6] = v_final
    return r

if __name__ == "__main__":
	kep = np.array([[15711.578566], [0.377617], [90.0], [0.887383], [0.0], [28.357744]])
	r = kep_state(kep)
	print(r)



