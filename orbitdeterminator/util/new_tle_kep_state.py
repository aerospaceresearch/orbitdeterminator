"""This module computes the state vector from keplerian elements."""

import math
import numpy as np
from scipy.optimize import fsolve

from orbitdeterminator.propagation.cowell import time_period

mu = 398600.4418

def __MtoE(M,e):
    """Calculates the eccentric anomaly from the mean anomaly.

       Args:
           M(float): the mean anomaly (in radians)
           e(float): the eccentricity

       Returns:
           float: The eccentric anomaly (in radians)
    """

    E = M
    dy = 1
    while(abs(dy) > 0.0001):
        M2 = E - e*math.sin(E)
        dy = M - M2
        dx = dy/(1-e*math.cos(E))
        E = E+dx

    return E

def __TtoE(T,e):
    E = math.atan2((1-e**2)**0.5*math.sin(T),e+math.cos(T))
    E = E%(2*math.pi)
    return E

def __EtoT(E,e):
    T = math.atan2((1-e**2)**0.5*math.sin(E),math.cos(E)-e)
    T = T%(2*math.pi)
    return T

def MtoT(M,e):
    return __EtoT(__MtoE(M,e),e)

def tle_to_state(tle):
    """ This function converts from TLE elements to position and velocity vector

        Args:
            tle(1x6 numpy array): tle contains the following variables
                tle[0] = inclination (degrees)
                tle[1] = right ascension of the ascending node (degrees)
                tle[2] = eccentricity (number)
                tle[3] = argument of perigee (degrees)
                tle[4] = mean anomaly (degrees)
                tle[5] = mean motion (revs per day)

        Returns:
        r: 1x6 numpy array which contains the position and velocity vector
           r[0],r[1],r[2] = position vector [rx,ry,rz] km
           r[3],r[4],r[5] = velocity vector [vx,vy,vz] km/s
    """

    # unload orbital elements array
    T = 86400/tle[5]
    sma = (mu*(T/2/math.pi)**2)**(1/3)
    ecc = tle[2]  # eccentricity
    inc = tle[0]  # inclination
    argp = tle[3]  # argument of perigee
    raan = tle[1]  # right ascension of the ascending node
    tanom = MtoT(math.radians(tle[4]), ecc)  # we use mean anomaly(kep(5)) and the function MtoT to compute true anomaly (tanom)
    tanom = math.degrees(tanom)%360

    kep = np.array([sma,ecc,inc,argp,raan,tanom])
    s = kep_to_state(kep)

    print("Keplerian Elements:")
    print(kep)

    return s

def kep_to_state(kep):
    """ This function converts from keplerian elements to the position and velocity vector

        Args:
            kep(1x6 numpy array): kep contains the following variables
                kep[0] = semi-major axis (kms)
                kep[1] = eccentricity (number)
                kep[2] = inclination (degrees)
                kep[3] = argument of perigee (degrees)
                kep[4] = right ascension of ascending node (degrees)
                kep[5] = true anomaly (degrees)

        Returns:
        r: 1x6 numpy array which contains the position and velocity vector
           r[0],r[1],r[2] = position vector [rx,ry,rz] km
           r[3],r[4],r[5] = velocity vector [vx,vy,vz] km/s
    """

    r = np.zeros((6,1))

    # unload orbital elements array

    sma = kep[0]  # sma is semi major axis
    ecc = kep[1]  # eccentricity
    inc = math.radians(kep[2])  # inclination
    argp = math.radians(kep[3])  # argument of perigee
    raan = math.radians(kep[4])  # right ascension of the ascending node
    eanom = __TtoE(math.radians(kep[5]), ecc)  # we use mean anomaly(kep(5)) and the function MtoE to compute eccentric anomaly (eanom)

    smb = sma * math.sqrt(1-ecc**2)

    x = sma * (math.cos(eanom) - ecc)
    y = smb * math.sin(eanom)

    # calculate position and velocity in orbital frame
    m_dot = (mu/sma**3)**0.5
    e_dot = m_dot/(1 - ecc*math.cos(eanom))
    x_dot = -sma * math.sin(eanom) * e_dot
    y_dot =  smb * math.cos(eanom) * e_dot

    # rotate them by argp degrees
    x_rot = x * math.cos(argp) - y * math.sin(argp)
    y_rot = x * math.sin(argp) + y * math.cos(argp)
    x_dot_rot = x_dot * math.cos(argp) - y_dot * math.sin(argp)
    y_dot_rot = x_dot * math.sin(argp) + y_dot * math.cos(argp)

    # convert them into 3D coordinates
    r[0] = x_rot * math.cos(raan) - y_rot * math.sin(raan) * math.cos(inc)
    r[1] = x_rot * math.sin(raan) + y_rot * math.cos(raan) * math.cos(inc)
    r[2] = y_rot * math.sin(inc)

    r[3] = x_dot_rot * math.cos(raan) - y_dot_rot * math.sin(raan) * math.cos(inc)
    r[4] = x_dot_rot * math.sin(raan) + y_dot_rot * math.cos(raan) * math.cos(inc)
    r[5] = y_dot_rot * math.sin(inc)

    return r

if __name__ == "__main__":

	#tle = np.array([51.6382,7.1114,0.0002893,211.1340,148.9642,15.568214688])
        #tle = np.array([51.6428, 291.0075, 0.0003411, 263.9950, 245.8448, 15.54009155])
        tle = np.array([51.6418, 266.6543, 0.0003456, 290.0933, 212.4518, 15.54021918])
        r = tle_to_state(tle)
        print(r)
