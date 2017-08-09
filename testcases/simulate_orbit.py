import math
import numpy as np
import random
import matplotlib.pylab as plt
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# physical constants
G = 6.67408 * 10**-11
m_Earth = 5.97237 * 10**24
mu = G * m_Earth
r_earth = 6371000.0

re = [0.0, 0.0 ,0.0]

def simulate(a, e, inclination, raan, argument_of_periapsis, true_anomaly):
    unixtime_start = time.time()

    print(unixtime_start, a, e, inclination, raan, argument_of_periapsis, true_anomaly)

    jitter = [0.0, 10000.0, 20000.0, 40000.0, 80000.0]
    f = []
    f_meta = []
    for i in range(len(jitter)):
        filename = "orbit_simulated_" + str(unixtime_start).split(".")[0] +"d"+ str(unixtime_start).split(".")[1][:3] + "_" + str(int(jitter[i]))
        f.append(open("data/track/" + filename + ".csv", "w"))
        f[i].write("time\tx\ty\tz\n")

        f_meta.append(open("data/meta/" + filename + "_meta.csv", "w"))
        f_meta[i].write("filename\tunixtime_start\tsatid\tjitter\ta\te\tinclination\traan\targument_of_periapsis\ttrue_anomaly\n")
        f_meta[i].write(filename + "\t" + str(unixtime_start) + "\t" + "0" + "\t" + str(jitter[i]) + "\t" + str(a) + "\t" + str(e) +
                        "\t" + str(inclination) + "\t" + str(raan) + "\t" + str(argument_of_periapsis) + "\t" + str(true_anomaly) + "\n")
        f_meta[i].close()

    argument_of_periapsis = argument_of_periapsis * np.pi / 180.0
    inclination = inclination* np.pi / 180.0
    raan = raan * np.pi / 180.0

    if e > 1.0: # hyperbel
        h = (a * mu * (e**2 - 1))**0.5
    elif e < 1.0: # elipsoid or circle
        h = (a * mu * (1 - e**2))**0.5

    r = [0,0,0]
    r[0] = h**2 / mu * 1/(1 + e * np.cos(true_anomaly * np.pi / 180.0)) * np.cos(true_anomaly * np.pi / 180.0)
    r[1] = h**2 / mu * 1/(1 + e * np.cos(true_anomaly * np.pi / 180.0)) * np.sin(true_anomaly * np.pi / 180.0)
    r[2] = h**2 / mu * 1/(1 + e * np.cos(true_anomaly * np.pi / 180.0)) * 0.0

    v = [0,0,0]
    v[0] = mu / h * -np.sin(true_anomaly * np.pi / 180.0)
    v[1] = mu / h * (e + np.cos(true_anomaly * np.pi / 180.0))
    v[2] = mu / h * 0.0


    A1 = np.array([[np.cos(argument_of_periapsis), np.sin(argument_of_periapsis), 0],
                 [-np.sin(argument_of_periapsis), np.cos(argument_of_periapsis), 0],
                 [0, 0, 1]])

    A2 = np.array([[1, 0, 0],
                 [0, np.cos(inclination), np.sin(inclination)],
                 [0, -np.sin(inclination), np.cos(inclination)]])

    A3 = np.array([[np.cos(raan), np.sin(raan), 0],
                 [-np.sin(raan), np.cos(raan), 0],
                 [0, 0, 1]])

    A = np.mat(A1) * np.mat(A2) * np.mat(A3)

    R = np.matmul(A.T, r)
    V = (np.matmul(A.T, v))

    #print(R[0,0])
    r0 = [R[0,0], R[0,1], R[0,2]]
    v0 = [V[0,0], V[0,1], V[0,2]]

    dt = 1.0

    a = [0,0,0]

    x = []
    y = []
    z = []

    x1 = []
    y1 = []
    z1 = []

    for t in range(0, 8000, 1):
        distance = ((r0[0] - re[0])**2 + (r0[1] - re[1])**2 + (r0[2] - re[2])**2)**0.5
        a[0] = G*m_Earth * (re[0] - r0[0]) / distance**3
        a[1] = G*m_Earth * (re[1] - r0[1]) / distance**3
        a[2] = G*m_Earth * (re[2] - r0[2]) / distance**3

        v0[0] = v0[0] + a[0] * dt
        v0[1] = v0[1] + a[1] * dt
        v0[2] = v0[2] + a[2] * dt

        r0[0] = r0[0] + v0[0] * dt
        r0[1] = r0[1] + v0[1] * dt
        r0[2] = r0[2] + v0[2] * dt

        #print(unixtime_start + t * dt, r0[0], r0[1], r0[2])
        for jit in range(len(jitter)):
            dx = random.uniform(-jitter[jit], jitter[jit])
            dy = random.uniform(-jitter[jit], jitter[jit])
            dz = random.uniform(-jitter[jit], jitter[jit])

            if jit == 1:
                x1.append(r0[0] + dx)
                y1.append(r0[1] + dy)
                z1.append(r0[2] + dz)

            f[jit].write(str(unixtime_start + t * dt) +
                         "\t" + str(r0[0] + dx) +
                         "\t" + str(r0[1] + dy) +
                         "\t" + str(r0[2] + dz) + "\n")

        x.append(r0[0])
        y.append(r0[1])
        z.append(r0[2])

    for jit in range(len(jitter)):
        f[jit].close()


    mpl.rcParams['legend.fontsize'] = 10

    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, "o", label='orbit original')
    ax.plot(x1, y1, z1, "*", label='orbit jitter')
    ax.legend()
    ax.set_xlim([-20000000, 20000000])
    ax.set_ylim([-20000000, 20000000])
    ax.set_zlim([-20000000, 20000000])
    '''
    plt.show()

if __name__ == '__main__':
    '''
    h = 80000000000.0
    a = r_earth + 400000.0
    e = 0.9

    inclination = 30.0
    raan = 40.0
    argument_of_periapsis = 60.0
    true_anomaly = 30.0
    '''

    for i in range(100):
        a = r_earth + random.uniform(200000.0, 1000000.0)
        e = random.uniform(0.0, 0.9999999999999)
        inclination = random.uniform(0.0, 180.0)
        raan = random.uniform(0.0, 360.0)
        argument_of_periapsis = random.uniform(0.0, 360.0)
        true_anomaly = random.uniform(0.0, 360.0)

        simulate(a, e, inclination, raan, argument_of_periapsis, true_anomaly)