import math
import numpy as np
import random
import matplotlib.pylab as plt
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# physical constants
mu = 398600.0
r = 6371.0

def newtons_algo(Me, eccentricity):
    # newton's algo
    # 1) starting with E0 guess
    E0 = 0.0
    if Me < np.pi:
        E0 = Me + eccentricity / 2.0
    else:
        E0 = Me - eccentricity / 2.0

    ratio = 1.0
    while np.abs(ratio) > 10**-8:
        # 2)
        fE = E0 - eccentricity * np.sin(E0) - Me
        ffE = 1.0 - eccentricity * np.cos(E0)

        ratio = fE / ffE
        #print(E0, fE, ffE, ratio)
        E0 = E0 - ratio

    return E0

if __name__ == '__main__':

    # orbital parameters
    ra = [21000.0]
    rp = [9600.0]
    argument_of_periapsis = [0 * np.pi / 180.0]
    inclination = [90 * np.pi / 180.0]
    raan = [0 * np.pi / 180.0]


    result = [[], []]
    r = np.array([[0.0,0.0,0.0]])
    v = np.array([[0,0,0]])


    x = [[], []]
    y = [[], []]
    z = [[], []]
    t = [[], []]

    for t1 in range(0, 18800, 100):
        t_orbit = t1

        for sat in range(len(ra)):
            eccentricity = (ra[sat] - rp[sat]) / (ra[sat] + rp[sat])
            h = (rp[sat] * (1.0 + eccentricity * np.cos(0)) * mu)**0.5
            period_orbit = 2.0 * np.pi / mu**2 * (h / (1 - eccentricity**2.0)**0.5)**3.0


            Me = 2 * np.pi * t_orbit / period_orbit

            E0 = newtons_algo(Me, eccentricity)

            true_anomaly = 2.0 * np.arctan(((1.0 + eccentricity) / (1.0 - eccentricity))**0.5 * np.tan(E0 / 2.0))

            if true_anomaly * 180.0 / np.pi >= 0.0:
                true_anomaly = true_anomaly * 180.0 / np.pi
            else:
                true_anomaly = true_anomaly * 180.0 / np.pi + 360

            #print(sat, t, t_orbit, Me, h, eccentricity, E0, eccentricity, true_anomaly)
            result[sat].append(true_anomaly)

            true_anomaly = true_anomaly * np.pi / 180.0

            r[sat][0] = h**2 / mu * 1.0 / (1.0 + eccentricity * np.cos(true_anomaly)) * np.cos(true_anomaly)
            r[sat][1] = h**2 / mu * 1.0 / (1.0 + eccentricity * np.cos(true_anomaly)) * np.sin(true_anomaly)

            v[sat][0] = mu/h * -np.sin(true_anomaly)
            v[sat][1] = mu/h * (eccentricity + np.cos(true_anomaly))

            A1 = np.array([[np.cos(argument_of_periapsis[sat]), np.sin(argument_of_periapsis[sat]), 0],
                 [-np.sin(argument_of_periapsis[sat]), np.cos(argument_of_periapsis[sat]), 0],
                 [0, 0, 1]])

            A2 = np.array([[1, 0, 0],
                 [0, np.cos(inclination[sat]), np.sin(inclination[sat])],
                 [0, -np.sin(inclination[sat]), np.cos(inclination[sat])]])

            A3 = np.array([[np.cos(raan[sat]), np.sin(raan[sat]), 0],
                 [-np.sin(raan[sat]), np.cos(raan[sat]), 0],
                 [0, 0, 1]])

            A = np.mat(A1) * np.mat(A2) * np.mat(A3)

            R = np.matmul(A.T, r[sat])
            V = (np.matmul(A.T, v[sat]))

            if sat == 0:
                print(R)

            x[sat].append(R[0,0])
            y[sat].append(R[0,1])
            z[sat].append(R[0,2])
            t[sat].append(t1)


    # output
    # 3d graph
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for sat in range(len(ra)):
        ax.plot(x[sat], y[sat], z[sat], "o", label='orbit original')
        f = open("orbit"+str(sat)+".dat", "w")
        for i in range(len(z[sat])):
            f.write(str(t[sat][i]) +","+ str(x[sat][i]) +","+ str(y[sat][i]) +","+ str(z[sat][i]) + "\n")
        f.close
    ax.legend()

    plt.show()
