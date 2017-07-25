import numpy as np
import math

def state_kep(r, v):
    ''' Converts state vector to orbital elements.

    Convert state vector to six classical orbital elements via equinoctial
     elements.

    Args:
        r (numpy array): position vector
        v (numpy array): velocity vector

    Returns:
        kep (numpy array): keplerian elements
        kep(0): semimajor axis (kilometers)
        kep(1): orbital eccentricity (non-dimensional)
                 (0 <= eccentricity < 1)
        kep(2): orbital inclination (degrees)
        kep(3): right ascension of ascending node (degrees)
        kep(4): argument of perigee (degress)
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

    true_anom = math.acos(np.dot(e, r) / (mag_r * mag_e))
    if np.dot(r, v) < 0:
        true_anom = 2 * math.pi - true_anom
    true_anom = math.degrees(true_anom)

    i = math.acos(h[2] / mag_h)
    i = math.degrees(i)

    ecc = mag_e

    raan = math.acos(n[0] / mag_n)
    if n[1] < 0:
        raan = 2 * math.pi - raan
    raan = math.degrees(raan)

    per = math.acos(np.dot(n, e) / (mag_n * mag_e))
    if e[2] < 0:
        per = 2 * math.pi - per
    per = math.degrees(per)

    a = 1 / ((2 / mag_r) - (mag_v**2 / mu))

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
