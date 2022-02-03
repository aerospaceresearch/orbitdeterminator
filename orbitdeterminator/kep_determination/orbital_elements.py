""" Implements simple keplerian orbital elements calculations. """

import numpy as np
from astropy import units as uts
from astropy import constants as cts

# declare astronomical constants in appropriate units
mu_Earth = cts.GM_earth.to(uts.Unit('km3 / s2')).value


def keplers_equation_by_newtons_method(eccentricity, mean_anomaly, etol=1.e-8):

    E_eccentric_anomaly = mean_anomaly - eccentricity / 2.0
    if mean_anomaly < np.pi:
        E_eccentric_anomaly = mean_anomaly + eccentricity / 2.0

    ratio = 1.0
    while np.abs(ratio) > etol:
        ratio = (E_eccentric_anomaly - eccentricity * np.sin(E_eccentric_anomaly) - mean_anomaly) / (1.0 - eccentricity * np.cos(E_eccentric_anomaly))
        E_eccentric_anomaly = E_eccentric_anomaly - ratio

    return E_eccentric_anomaly


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


def h_angularmomentuum(R=None, V=None, semimajor_axis=None, eccentricity=None):

    h_angularmomentuum = None

    if R.any != None and V.any != None:
        h = np.cross(R, V)
        h_angularmomentuum = np.linalg.norm(h)

    if semimajor_axis != None and eccentricity != None:
        h_angularmomentuum = (semimajor_axis * mu_Earth * (1.0 - eccentricity**2))**0.5

    return h_angularmomentuum


def inclination(R, V):

    h = np.cross(R, V)
    h_angularmomentuum_abs = h_angularmomentuum(R=R, V=V)

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


def true_anomaly(R=None, V=None, eccentricity=None, E_eccentric_anomaly=None):

    true_anomaly = None

    if R.any != None and V.any != None:

        r_abs = np.linalg.norm(R)
        v_abs = np.linalg.norm(V)

        v_radial = np.dot(R, V) / r_abs

        eccentricity_vec = eccentricity_v(R, V)
        eccentricity = np.linalg.norm(eccentricity_vec)

        true_anomaly = np.arccos(np.dot(eccentricity_vec / eccentricity, R / r_abs))

        if v_radial < 0.0:
            true_anomaly = 2.0 * np.pi - true_anomaly


    if eccentricity != None and E_eccentric_anomaly != None:
        true_anomaly = 2.0 * np.arctan(((1.0 + eccentricity) / (1.0 - eccentricity)) ** 0.5 *
                                       np.tan(E_eccentric_anomaly / 2.0))

        true_anomaly = zeroTo360(true_anomaly, deg=False)

    return true_anomaly


def E_eccentric_anomaly(eccentricity=None, true_anomaly=None, mean_anomaly=None):

    E_eccentric_anomaly = None

    if eccentricity != None and true_anomaly != None:
        E_eccentric_anomaly = 2.0 * np.arctan(((1.0 - eccentricity) / (1.0 + eccentricity)) ** 0.5
                                              * np.tan(true_anomaly / 2.0))

    if eccentricity != None and mean_anomaly != None:
        E_eccentric_anomaly = keplers_equation_by_newtons_method(eccentricity, mean_anomaly)

    return E_eccentric_anomaly


def mean_anomaly(E_eccentric_anomaly, eccentricity, true_anomaly):

    mean_anomaly = E_eccentric_anomaly - eccentricity * np.sin(E_eccentric_anomaly)

    if eccentricity > 1.0:
        # if hyperbola
        F_hyp_eccentric_anomaly = 2.0 * np.arctanh(((eccentricity - 1.0) / (eccentricity + 1.0)) ** 0.5
                                                   * np.tan(true_anomaly / 2.0))

        mean_anomaly = eccentricity * np.sinh(F_hyp_eccentric_anomaly) - F_hyp_eccentric_anomaly

    mean_anomaly = zeroTo360(mean_anomaly, deg=False)

    return mean_anomaly


def T_orbitperiod(h_angularmomentuum = None, eccentricity = None, semimajor_axis = None, n_mean_motion_perday=None):

    T_orbitperiod = None

    if h_angularmomentuum != None and eccentricity != None:
        T_orbitperiod = 2.0 * np.pi / mu_Earth ** 2 * (h_angularmomentuum / np.sqrt(1 - eccentricity ** 2)) ** 3

    if semimajor_axis != None:
        T_orbitperiod = 2.0 * np.pi / mu_Earth ** 0.5 * semimajor_axis ** (3.0/2.0)

    if n_mean_motion_perday != None:
        T_orbitperiod = (24.0 * 3600.0) / n_mean_motion_perday

    return T_orbitperiod


def semimajor_axis(R = None, V = None, T_orbitperiod = None):

    semimajor_axis = None

    if R.any != None and V.any != None:
        eccentricity_vec = eccentricity_v(R, V)
        eccentricity = np.linalg.norm(eccentricity_vec)

        h_abs = h_angularmomentuum(R=R, V=V)

        semimajor_axis = h_abs ** 2 / mu_Earth * 1.0 / (1.0 - eccentricity ** 2)

        if eccentricity > 1.0:
            # if hyperbola
            semimajor_axis = h_abs ** 2 / mu_Earth * 1.0 / (eccentricity ** 2 - 1.0)

    if T_orbitperiod != None:
        semimajor_axis = (T_orbitperiod * mu_Earth ** 0.5 / (2.0 * np.pi)) ** (2.0 / 3.0)

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
        self.R = []
        self.V = []


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

        self.true_anomaly = true_anomaly(R=R, V=V)

        self.E_eccentric_anomaly = E_eccentric_anomaly(self.eccentricity, self.true_anomaly)

        self.mean_anomaly = mean_anomaly(self.E_eccentric_anomaly, self.eccentricity, self.true_anomaly)

        self.T_orbitperiod = T_orbitperiod(self.h_angularmomentuum, self.eccentricity)

        self.t_since_perigee = t_since_perige(self.mean_anomaly, self.T_orbitperiod)

        self.n_mean_motion = n_mean_motion(self.T_orbitperiod)
        self.n_mean_motion_perday = n_mean_motion_perday(self.T_orbitperiod)

        self.semimajor_axis = semimajor_axis(R, V)

        self.p_orbitparameter = p_orbitparameter(self.h_angularmomentuum)


    def get_statevector_from_orbital_elemts(self, inclination, raan, eccentricity,
                                            AoP, mean_anomaly, n_mean_motion_perday):

        self.inclination = inclination
        self.raan = raan
        self.eccentricity = eccentricity
        self.AoP = AoP
        self.mean_anomaly = mean_anomaly
        self.n_mean_motion_perday = n_mean_motion_perday

        self.T_orbitperiod = T_orbitperiod(n_mean_motion_perday = self.n_mean_motion_perday)
        self.semimajor_axis = semimajor_axis(T_orbitperiod=self.T_orbitperiod)
        self.h_angularmomentuum = h_angularmomentuum(semimajor_axis=self.semimajor_axis, eccentricity=self.eccentricity)
        self.E_eccentric_anomaly = E_eccentric_anomaly(eccentricity=self.eccentricity,
                                                       mean_anomaly=self.mean_anomaly * np.pi / 180.0)
        self.true_anomaly = true_anomaly(eccentricity=self.eccentricity, E_eccentric_anomaly=self.E_eccentric_anomaly) * 180. / np.pi

        # p. 211
        R = self.h_angularmomentuum ** 2 / mu_Earth * \
            1.0 / (1 + self.eccentricity * np.cos(self.true_anomaly * np.pi / 180.0))
        R = R * np.array([np.cos(self.true_anomaly * np.pi / 180.0),
                          np.sin(self.true_anomaly * np.pi / 180.0),
                          0.0])

        V = mu_Earth / self.h_angularmomentuum
        V = V * np.array([-np.sin(self.true_anomaly * np.pi / 180.0),
                          self.eccentricity + np.cos(self.true_anomaly * np.pi / 180.0),
                          0.0])

        inc = self.inclination * np.pi / 180
        raan = self.raan * np.pi / 180
        aop = self.AoP * np.pi / 180

        Q_Xx = np.array([[+np.cos(aop), np.sin(aop), 0.0],
                         [-np.sin(aop), np.cos(aop), 0.0],
                         [0.0, 0.0, 1.0]])
        Q_Xx = np.dot(Q_Xx,
                      np.array([[1.0, 0.0, 0.0],
                                [0.0, +np.cos(inc), np.sin(inc)],
                                [0.0, -np.sin(inc), np.cos(inc)]]))
        Q_Xx = np.dot(Q_Xx,
                      np.array([[+np.cos(raan), np.sin(raan), 0.0],
                                [-np.sin(raan), np.cos(raan), 0.0],
                                [0.0, 0.0, 1.0]]))

        # Q_xX = np.transpose(Q_Xx)
        self.R = np.dot(R, Q_Xx)
        self.V = np.dot(V, Q_Xx)

if __name__ == "__main__":
    '''
            OMFES (4th), H.D.Curtis, p.210, ALGORITHM 4.5

            :param R: position vector of satellite [km]
            :param V: velocity vector of satellite [km/s
            :return:
            '''

    from astropy.time import Time
    from skyfield.api import load, wgs84
    from skyfield.api import EarthSatellite
    import time

    line1 = "1 47856U 21020C   21331.00000000 -.00000072  00000-0  30707-4 0  9998"
    line2 = "2 47856  63.4086 209.7279 0025693 342.6358  17.3780 13.45215752 34829"

    inclination = 63.4086
    raan = 209.7279
    eccentricity = 0.0025693
    AoP = 342.6358
    mean_anomaly = 17.3780
    n_mean_motion_perday = 13.45215752

    epoch = 21331.00000000 #2021-11-27T00:00:00

    observing_time = Time("2021-11-27T00:00:00", format="isot", scale="utc")
    ts = load.timescale()
    t = ts.from_astropy(observing_time)
    satellite = EarthSatellite(line1, line2, 'TEST', ts)
    R1 = satellite.at(t).position.km
    print(R1, np.linalg.norm(R1))


    oe = orbital_parameters()
    oe.get_statevector_from_orbital_elemts(inclination, raan, eccentricity,
                                            AoP, mean_anomaly, n_mean_motion_perday)

    R2 = oe.R
    V2 = oe.V

    print(R2, np.linalg.norm(R2))
    print(np.linalg.norm(R2-R1))