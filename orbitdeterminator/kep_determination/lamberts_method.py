""" Implements lambert's method for two topocentric radius vectors at different times. Supports both Earth-centered."""


import numpy as np
from poliastro.core.stumpff import c2, c3
from astropy import units as uts
from astropy import constants as cts

# declare astronomical constants in appropriate units
mu_Earth = cts.GM_earth.to(uts.Unit('km3 / s2')).value


def F_z_i(z, t, r1, r2, A):
    """ Function F for Newton's method

    :param z:
    :param t:
    :param r1:
    :param r2:
    :param A:
    :return:

    F: function

    """
    # mu = mu_Earth
    C_z_i = c2(z)
    S_z_i = c3(z)
    y_z = r1 + r2 + A * (z * S_z_i - 1.0) / np.sqrt(C_z_i)

    F = (y_z / C_z_i) ** 1.5 * S_z_i + A * np.sqrt(np.abs(y_z)) - np.sqrt(mu_Earth) * t

    return F


def dFdz(z, r1, r2, A):
    """ Derivative of Function F for Netwon's Method

    :param z:
    :param r1:
    :param r2:
    :param A:
    :return:

    df: derivative for Netwon's method.

    """
    # mu = mu_Earth

    if z == 0:
        C_z_i = c2(0)
        S_z_i = c3(0)
        y_0 = r1 + r2 + A * (0 * S_z_i - 1.0) / np.sqrt(C_z_i)

        dF = np.sqrt(2) / 40.0 * y_0**1.5 + A / 8.0 * (np.sqrt(y_0) + A * np.sqrt(1 / 2 / y_0))

    else:
        C_z_i = c2(z)
        S_z_i = c3(z)
        y_z = r1 + r2 + A * (z * S_z_i - 1.0) / np.sqrt(C_z_i)

        dF = (y_z / C_z_i)**1.5 * \
             (1.0 / 2.0 / z * (C_z_i - 3.0 * S_z_i/ 2.0 / C_z_i) + 3.0 * S_z_i + 2.0 / 4.0 / C_z_i) +\
             A / 8.0 * (3.0 * S_z_i / C_z_i * np.sqrt(y_z) +  A * np.sqrt(C_z_i / y_z))

    return dF


def lamberts_method(R1, R2, delta_time, trajectory_cw=False):
    """ lamberts method that generates velocity vectors for each radius vectors that are put in here.

    Similar to https://esa.github.io/pykep/documentation/core.html#pykep.lambert_problem

    :param R1: radius vector of measuements #1
    :param R2: radius vector of measuements #2
    :param delta_time: delta time between R1 and R2 measurements, in seconds. Only works for same pass
    :param trajectory_cw: bool. True for retrograde motion (clockwise), False if counter-clock wise
    :return:

    V1: velocity vector of R1
    V2: velocity vector of R2
    ratio: absolute tolerance result

    """

    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)

    c12 = np.cross(R1, R2)

    theta = np.arccos(np.dot(R1, R2) / r1 / r2)

    # todo: automatic check for pro or retrograde orbits is needed. currently set to prograde
    if trajectory_cw == False:
        trajectory = "prograde"
    if trajectory_cw == True:
        trajectory = "retrograde"

    #inclination = 0.0
    #if inclination >= 0.0 and inclination < 90.0:
    if trajectory == "prograde":
        if c12[2] >= 0:
            theta = theta
        else:
            theta = np.pi*2 - theta

    #if inclination >= 90.0 and inclination < 180.0:
    if trajectory == "retrograde":
        if c12[2] < 0:
            theta = theta
        else:
            theta = np.pi*2 - theta

    A = np.sin(theta) * np.sqrt(r1 * r2 / (1.0 - np.cos(theta)))

    z = 0.0
    while F_z_i(z, delta_time, r1, r2, A) < 0.0:
        z = z + 0.1

    tol = 1.e-10 # tolerance
    nmax = 6000 # iterations

    ratio = 1
    n = 0
    while (np.abs(ratio) > tol) and (n <= nmax):
        n = n + 1
        ratio = F_z_i(z, delta_time, r1, r2, A)

        if np.isnan(ratio) == True:
            break
        ratio = ratio / dFdz(z, r1, r2, A)
        z = np.abs(z - ratio)


    C_z_i = c2(z)
    S_z_i = c3(z)
    y_z = r1 + r2 + A * (z * S_z_i - 1.0) / np.sqrt(C_z_i)

    f = 1.0 - y_z / r1

    # mu = mu_Earth
    g = A * np.sqrt(y_z / mu_Earth)

    gdot = 1.0 - y_z / r2

    V1 = 1.0 / g * (np.add(R2, -np.multiply(f, R1)))
    V2 = 1.0 / g * (np.multiply(gdot, R2) - R1)

    return V1, V2, np.abs(ratio)
