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


    def get_orbital_elemts_from_statevector(self, R, V):
        '''
        OMFES (4th), H.D.Curtis, p.191, ALGORITHM 4.2

        :param R: position vector of satellite [km]
        :param V: velocity vector of satellite [km/s
        :return:
        '''

        # todo: if other body, this shall be changed
        mu = mu_Earth


        r_abs = np.linalg.norm(R)
        v_abs = np.linalg.norm(V)

        self.v_radial = np.dot(R,V) / r_abs

        h = np.cross(R, V)
        self.h_angularmomentuum = np.linalg.norm(h)

        self.inclination = np.arccos(h[2] / self.h_angularmomentuum)

        node_line = np.cross([0.0, 0.0, 1.0], h)
        node_line_abs = np.linalg.norm(node_line)

        self.raan = np.arccos(node_line[0] / node_line_abs)
        if node_line[1] < 0.0:
            self.raan = 2.0 * np.pi - self.raan


        eccentricity = 1.0 / mu * ((v_abs ** 2 - mu / r_abs) * R - r_abs * self.v_radial * V)
        self.eccentricity = np.linalg.norm(eccentricity)

        self.AoP = np.arccos(np.dot(node_line / node_line_abs,  eccentricity / self.eccentricity))
        if eccentricity[2] < 0.0:
            self.AoP = 2.0 * np.pi - self.AoP

        self.true_anomaly = np.arccos(np.dot(eccentricity / self.eccentricity, R / r_abs))
        if self.v_radial < 0.0:
            self.true_anomaly = 2.0 * np.pi - self.true_anomaly


        E_eccentric_anomaly = 2.0 * np.arctan(((1.0 - self.eccentricity) / (1.0 + self.eccentricity))**0.5
                                            * np.tan(self.true_anomaly / 2.0))

        self.mean_anomaly = E_eccentric_anomaly - self.eccentricity * np.sin(E_eccentric_anomaly)

        if self.eccentricity > 1.0:
            # if hyperbola
            F_hyp_eccentric_anomaly = 2.0 * np.arctanh(((self.eccentricity - 1.0) / (self.eccentricity + 1.0))**0.5
                                            * np.tan(self.true_anomaly / 2.0))
            self.mean_anomaly = self.eccentricity * np.sinh(F_hyp_eccentric_anomaly) - F_hyp_eccentric_anomaly

        self.mean_anomaly = zeroTo360(self.mean_anomaly, False)


        self.T_orbitperiod = 2.0 * np.pi / mu ** 2 * (self.h_angularmomentuum / np.sqrt(1 - self.eccentricity ** 2)) ** 3

        self.t_since_perigee = self.mean_anomaly / (2.0 * np.pi) * self.T_orbitperiod

        self.n_mean_motion = (2.0 * np.pi) / self.T_orbitperiod
        self.n_mean_motion_perday = 24.0 * 3600.0 / self.T_orbitperiod

        self.semimajor_axis = self.h_angularmomentuum ** 2 / mu * 1.0 / (1.0 - self.eccentricity ** 2)
        if self.eccentricity > 1.0:
            # if hyperbola
            self.semimajor_axis = self.h_angularmomentuum ** 2 / mu * 1.0 / (self.eccentricity ** 2 - 1.0)

        self.p_orbitparameter = self.h_angularmomentuum ** 2 / mu