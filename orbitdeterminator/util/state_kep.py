'''
Takes a state vector (x, y, z, vx, vy, vz) where v is the velocity of the satellite
and transforms it into a set of keplerian elements (a, e, i, ω, Ω, v)

'''

import numpy as np
import math

def state_kep(r, v):
    '''
    Converts state vector to orbital elements.

    Args:
        r (numpy array): position vector
        v (numpy array): velocity vector

    Returns:
        numpy array: array of the computed keplerian elements
        kep(0): semimajor axis (kilometers)
        kep(1): orbital eccentricity (non-dimensional)
                 (0 <= eccentricity < 1)
        kep(2): orbital inclination (degrees)
        kep(3): argument of perigee (degress)
        kep(4): right ascension of ascending node (degrees)
        kep(5): true anomaly (degrees)
    '''

    mu = 398600.4405
    mag_r = np.sqrt(r.dot(r))
    mag_v = np.sqrt(v.dot(v))

    h = np.cross(r, v)
    mag_h = np.sqrt(h.dot(h))

    e = ((np.cross(v, h)) / mu) - (r / mag_r)
    mag_e = np.sqrt(e.dot(e))

    n = np.array([-h[1], h[0], 0])
    mag_n = np.sqrt(n.dot(n))

    true_anom = math.acos(np.clip(np.dot(e,r)/(mag_r * mag_e), -1, 1))
    if np.dot(r, v) < 0:
        true_anom = 2 * math.pi - true_anom
    true_anom = math.degrees(true_anom)

    i = math.acos(np.clip(h[2] / mag_h, -1, 1))
    i = math.degrees(i)

    ecc = mag_e

    raan = math.acos(np.clip(n[0] / mag_n, -1, 1))
    if n[1] < 0:
        raan = 2 * math.pi - raan
    raan = math.degrees(raan)

    per = math.acos(np.clip(np.dot(n, e) / (mag_n * mag_e), -1, 1))
    if e[2] < 0:
        per = 2 * math.pi - per
    per = math.degrees(per)

    a = 1 / ((2 / mag_r) - (mag_v**2 / mu))

    if i >= 360.0:
        i = i - 360
    if raan >= 360.0:
        raan = raan - 360
    if per >= 360.0:
        per = per - 360

    kep = np.zeros(6)
    kep[0] = a
    kep[1] = ecc
    kep[2] = i
    kep[3] = per
    kep[4] = raan
    kep[5] = true_anom
    return kep


if __name__ == "__main__":
    r = np.array([5.0756899358316559e+03, -4.5590381308371752e+03, 1.9322228177731663e+03])
    v = np.array([1.3360847905126974e+00, -1.5698574946888049e+00, -7.2117328822023676e+00])

    kep = state_kep(r, v)
    print(kep)
