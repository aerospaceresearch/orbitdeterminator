""" Implements simple keplerian orbital elements calculations. """

import numpy as np
from astropy import units as uts
from astropy import constants as cts

# declare astronomical constants in appropriate units
mu_Earth = cts.GM_earth.to(uts.Unit('km3 / s2')).value


def zeroTo360(x, deg=True):

    if deg == True:
        base = 360.0
    else:
        base = 2.0 * np.pi


    if x >= base:
        x = x - int(x/base) * base

    elif x < 0.0:
        x = x - (int(x/base) - 1.0) * base

    return x


def v_radial(R, V):

    r_abs = np.linalg.norm(R)
    v_radial = np.dot(R, V) / r_abs

    return v_radial


def h_angularmomentuum(R, V):

    h = np.cross(R, V)
    h_angularmomentuum = np.linalg.norm(h)

    return h_angularmomentuum


def inclination(R, V):

    h = np.cross(R, V)
    h_angularmomentuum_abs = h_angularmomentuum(R, V)

    inclination = np.arccos(h[2] / h_angularmomentuum_abs)

    return inclination


def raan(R, V):

    h = np.cross(R, V)

    node_line = np.cross([0.0, 0.0, 1.0], h)
    node_line_abs = np.linalg.norm(node_line)

    raan = np.arccos(node_line[0] / node_line_abs)

    if node_line[1] < 0.0:
        raan = 2.0 * np.pi - raan

    return raan


def eccentricity_v(R, V):
    r_abs = np.linalg.norm(R)
    v_abs = np.linalg.norm(V)

    v_radial = np.dot(R, V) / r_abs

    eccentricity_vec = 1.0 / mu_Earth * ((v_abs ** 2 - mu_Earth / r_abs) * R - r_abs * v_radial * V)

    return eccentricity_vec


def AoP(R, V):

    h = np.cross(R, V)

    node_line = np.cross([0.0, 0.0, 1.0], h)
    node_line_abs = np.linalg.norm(node_line)

    eccentricity_vec = eccentricity_v(R, V)
    eccentricity = np.linalg.norm(eccentricity_vec)

    AoP = np.arccos(np.dot(node_line / node_line_abs, eccentricity_vec / eccentricity))

    if eccentricity_vec[2] < 0.0:
        AoP = 2.0 * np.pi - AoP

    return AoP


def true_anomaly(R , V):

    r_abs = np.linalg.norm(R)
    v_abs = np.linalg.norm(V)

    v_radial = np.dot(R, V) / r_abs

    eccentricity_vec = eccentricity_v(R, V)
    eccentricity = np.linalg.norm(eccentricity_vec)

    true_anomaly = np.arccos(np.dot(eccentricity_vec / eccentricity, R / r_abs))

    if v_radial < 0.0:
        true_anomaly = 2.0 * np.pi - true_anomaly

    return true_anomaly


def E_eccentric_anomaly(eccentricity, true_anomaly):

    E_eccentric_anomaly = 2.0 * np.arctan(((1.0 - eccentricity) / (1.0 + eccentricity)) ** 0.5
                                          * np.tan(true_anomaly / 2.0))

    return E_eccentric_anomaly


def mean_anomaly(E_eccentric_anomaly, eccentricity, true_anomaly):

    mean_anomaly = E_eccentric_anomaly - eccentricity * np.sin(E_eccentric_anomaly)

    if eccentricity > 1.0:
        # if hyperbola
        F_hyp_eccentric_anomaly = 2.0 * np.arctanh(((eccentricity - 1.0) / (eccentricity + 1.0)) ** 0.5
                                                   * np.tan(true_anomaly / 2.0))

        mean_anomaly = eccentricity * np.sinh(F_hyp_eccentric_anomaly) - F_hyp_eccentric_anomaly

    mean_anomaly = zeroTo360(mean_anomaly, False)

    return mean_anomaly


def T_orbitperiod(h_angularmomentuum = None, eccentricity = None, semimajor_axis = None):

    if h_angularmomentuum != None and eccentricity != None:
        T_orbitperiod = 2.0 * np.pi / mu_Earth ** 2 * (h_angularmomentuum / np.sqrt(1 - eccentricity ** 2)) ** 3

    if semimajor_axis != None:
        T_orbitperiod = 2.0 * np.pi / mu_Earth ** 0.5 * semimajor_axis * 3.0/2.0

    return T_orbitperiod


def semimajor_axis(R, V):

    eccentricity_vec = eccentricity_v(R, V)
    eccentricity = np.linalg.norm(eccentricity_vec)

    h_abs = h_angularmomentuum(R, V)

    semimajor_axis = h_abs ** 2 / mu_Earth * 1.0 / (1.0 - eccentricity ** 2)

    if eccentricity > 1.0:
        # if hyperbola
        semimajor_axis = h_abs ** 2 / mu_Earth * 1.0 / (eccentricity ** 2 - 1.0)

    return semimajor_axis


def t_since_perige(mean_anomaly, T_orbitperiod):

    t_since_perige = mean_anomaly / (2.0 * np.pi) * T_orbitperiod

    return t_since_perige


def p_orbitparameter(h_angularmomentuum):

    return h_angularmomentuum ** 2 / mu_Earth


def n_mean_motion(T_orbitperiod):

    return (2.0 * np.pi) / T_orbitperiod


def n_mean_motion_perday(T_orbitperiod):

    return 24.0 * 3600.0 / T_orbitperiod


class orbital_parameters:


    # The init method or constructor
    def __init__(self):

        self.AoP = 0.0
        self.true_anomaly = 0.0
        self.raan = 0.0
        self.eccentricity = 0.0
        self.mean_anomaly = 0.0
        self.T_orbitperiod = 0.0
        self.t_since_perigee = 0.0
        self.n_mean_motion = 0.0
        self.n_mean_motion_perday = 0.0
        self.semimajor_axis = 0.0
        self.p_orbitparameter = 0.0
        self.inclination = 0.0
        self.h_angularmomentuum = 0.0
        self.v_radial = 0.0
        self.E_eccentric_anomaly = 0.0


    def get_orbital_elemts_from_statevector(self, R, V):
        '''
        OMFES (4th), H.D.Curtis, p.191, ALGORITHM 4.2

        :param R: position vector of satellite [km]
        :param V: velocity vector of satellite [km/s
        :return:
        '''

        # todo: if other body, this shall be changed
        mu = mu_Earth

        self.v_radial = v_radial(R, V)

        self.h_angularmomentuum = h_angularmomentuum(R, V)

        self.inclination = inclination(R, V)

        self.raan = raan(R, V)

        eccentricity_vec = eccentricity_v(R, V)
        self.eccentricity = np.linalg.norm(eccentricity_vec)

        self.AoP = AoP(R, V)

        self.true_anomaly = true_anomaly(R, V)

        self.E_eccentric_anomaly = E_eccentric_anomaly(self.eccentricity, self.true_anomaly)

        self.mean_anomaly = mean_anomaly(self.E_eccentric_anomaly, self.eccentricity, self.true_anomaly)

        self.T_orbitperiod = T_orbitperiod(self.h_angularmomentuum, self.eccentricity)

        self.t_since_perigee = t_since_perige(self.mean_anomaly, self.T_orbitperiod)

        self.n_mean_motion = n_mean_motion(self.T_orbitperiod)
        self.n_mean_motion_perday = n_mean_motion_perday(self.T_orbitperiod)

        self.semimajor_axis = semimajor_axis(R, V)

        self.p_orbitparameter = p_orbitparameter(self.h_angularmomentuum)