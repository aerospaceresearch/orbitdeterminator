import numpy as np


# constants
mu = 398600.0


def get_state_vector(radius_apoapsis, radius_periapsis, inclincation, raan, AoP, time_after_periapsis):
    """
    simple state vector determination for a satellite in orbit.
    based on "Orbital Mechanics for Engineering Students" by Howard D. Curtis


    :param radius_apoapsis: [km] distance from elipsis center to apoapsis
    :param radius_periapsis: [km] distance from elipsis center to periapsis
    :param inclination: [°] inclination of the orbit
    :param raan: [°] right ascension of the ascending node
    :param AoP: [°] argument of periapsis
    :param time_after_periapsis: [s] time after periapsis passage

    :return: returns vectors for radius and velocities in km and km/s
    """

    eccentricity = (radius_apoapsis - radius_periapsis) / (radius_apoapsis + radius_periapsis)
    h_angularmomentuum = np.sqrt(radius_periapsis * (1 + eccentricity * np.cos(0)) * mu)
    T_orbitperiod = 2.0 * np.pi / mu**2 * (h_angularmomentuum / np.sqrt(1 - eccentricity**2))**3


    # newton method to find the true anomaly for a given time after perapsis passage
    Me = 2.0 * np.pi * time_after_periapsis / T_orbitperiod

    if Me > np.pi:
        E0 = Me - eccentricity / 2
    else:
        E0 = Me + eccentricity / 2


    ratioi = 1.0
    while np.abs(ratioi) > 10E-8:
        f_Ei = E0 - eccentricity * np.sin(E0) - Me
        f__Ei = 1 - eccentricity * np.cos(E0)

        ratioi = f_Ei / f__Ei

        E0 = E0 - ratioi


    true_anomaly = np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(E0 / 2)
    true_anomaly = np.arctan(true_anomaly) * 2.0 * 180.0 / np.pi

    if true_anomaly < 0.0:
        true_anomaly = true_anomaly + 360.0


    # degrees to rad again
    inclincation = inclincation * np.pi / 180.0
    raan = raan * np.pi / 180.0
    AoP = AoP * np.pi / 180.0
    true_anomaly = true_anomaly * np.pi / 180.0


    # determination of radius vector on plane of the elipsis
    r = h_angularmomentuum**2 / mu * 1.0 / (1.0 + eccentricity * np.cos(true_anomaly))
    r = np.multiply(r, [np.cos(true_anomaly), np.sin(true_anomaly), 0.0])

    # determination of velocity vector on plane of the elipsis
    v = mu / h_angularmomentuum
    v = np.multiply(v, [-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0])


    # stepwise creation of the coordination transform matrix
    Q_Xx = np.array([[np.cos(AoP), np.sin(AoP), 0.0],
                     [-np.sin(AoP), np.cos(AoP), 0.0],
                     [0.0, 0.0, 1.0]])

    Q_Xx = Q_Xx.dot(np.array([[1.0, 0.0, 0.0],
                              [0.0, np.cos(inclincation), np.sin(inclincation)],
                              [0.0, -np.sin(inclincation), np.cos(inclincation)]]))
    
    Q_Xx = Q_Xx.dot(np.array([[np.cos(raan), np.sin(raan), 0.0],
                              [-np.sin(raan), np.cos(raan), 0.0],
                              [0.0, 0.0, 1.0]]))


    # transforming the radius vectors and velocities into the 3d reference frame.
    R = Q_Xx.T.dot(r)
    V = Q_Xx.T.dot(v)

    return R, V