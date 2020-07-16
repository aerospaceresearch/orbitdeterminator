import numpy as np
from poliastro.core.stumpff import c2, c3
from astropy import units as uts
from astropy import constants as cts

# declare astronomical constants in appropriate units
mu_Earth = cts.GM_earth.to(uts.Unit('km3 / s2')).value


def F_z_i(z, t, r1, r2, A):
    mu = mu_Earth
    C_z_i = c2(z)
    S_z_i = c3(z)
    y_z = r1 + r2 + A * (z * S_z_i - 1.0) / np.sqrt(C_z_i)
    #print("qqq",z, y_z, C_z_i, S_z_i)
    F = (y_z / C_z_i) ** 1.5 * S_z_i + A * np.sqrt(np.abs(y_z)) - np.sqrt(mu) * t

    return F


def dFdz(z, r1, r2, A):
    mu = mu_Earth

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


def lamberts_method(R1, R2, delta_time):
    r1 = np.linalg.norm(R1)
    r2 = np.linalg.norm(R2)

    c12 = np.cross(R1, R2)

    theta = np.arccos(np.dot(R1, R2) / r1 / r2)

    trajectory = "prograde"
    inclination = 0.0
    if inclination >= 0.0 and inclination < 90.0:
        trajectory = "prograde"
        if c12[2] >= 0:
            theta = theta
        else:
            theta = np.pi*2 - theta

    if inclination >= 90.0 and inclination < 180.0:
        trajectory = "retrograde"
        if c12[2] < 0:
            theta = theta
        else:
            theta = np.pi*2 - theta

    A = np.sin(theta) * np.sqrt(r1 * r2 / (1.0 - np.cos(theta)))

    z = 0.0
    while F_z_i(z, delta_time, r1, r2, A) < 0.0:
        z = z + 0.1

    tol = 1.e-8
    nmax = 5000

    ratio = 1
    n = 0
    while (np.abs(ratio) > tol) and (n <= nmax):
        n = n + 1
        ratio = F_z_i(z, delta_time, r1, r2, A)
        #print(n, z, ratio)
        if np.isnan(ratio) == True:
            break
        ratio = ratio / dFdz(z, r1, r2, A)
        z = np.abs(z - ratio)


    C_z_i = c2(z)
    S_z_i = c3(z)
    y_z = r1 + r2 + A * (z * S_z_i - 1.0) / np.sqrt(C_z_i)

    f = 1.0 - y_z / r1

    mu = mu_Earth
    g = A * np.sqrt(y_z / mu)

    gdot = 1.0 - y_z / r2

    V1 = 1.0 / g * (np.add(R2, -np.multiply(f, R1)))
    V2 = 1.0 / g * (np.multiply(gdot, R2) - R1)

    return V1, V2