""" Implements simple keplerian orbital elements calculations. """

import numpy as np
from astropy import units as uts
from astropy import constants as cts

# declare astronomical constants in appropriate units
mu_Earth = cts.GM_earth.to(uts.Unit('km3 / s2')).value


def get_orbital_elemts_from_statevector(R, V):
    '''
    OMFES (4th), H.D.Curtis, p.191, ALGORITHM 4.2

    :param R: position vector of satellite [km]
    :param V: velocity vector of satellite [km/s
    :return:
    '''

    r_abs = np.linalg.norm(R)
    v_abs = np.linalg.norm(V)

    v_radial = np.dot(R,V) / r_abs

    h = np.cross(R, V)
    h_abs = np.linalg.norm(h)

    inclination = np.arccos(h[2] / h_abs)

    node_line = np.cross([0.0, 0.0, 1.0], h)
    node_line_abs = np.linalg.norm(node_line)

    raan = np.arccos(node_line[0] / node_line_abs)
    if node_line[1] < 0.0:
        raan = 2.0 * np.pi - raan


    mu = mu_Earth
    eccentricity = 1.0 / mu * ((v_abs ** 2 - mu / r_abs) * R - r_abs * v_radial * V)
    eccentricity_abs = np.linalg.norm(eccentricity)


    AoP = np.arccos(np.dot(node_line / node_line_abs,  eccentricity / eccentricity_abs))
    if eccentricity[2] < 0.0:
        AoP = 2.0 * np.pi - AoP

    true_anomaly = np.arccos(np.dot(eccentricity / eccentricity_abs, R / r_abs))
    if v_radial < 0.0:
        true_anomaly = 2.0 * np.pi - true_anomaly


    return inclination * 180.0 / np.pi, raan * 180.0 / np.pi, true_anomaly * 180.0 / np.pi, AoP * 180.0 / np.pi, eccentricity_abs, h_abs