# -*- coding: utf-8 -*-

"""
Code is taken from https://github.com/Elucidation/OrbitalElements
"""

from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
cos = np.cos
sin = np.sin
pi = np.pi
dot = np.dot

fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d', aspect=1)

max_radius = 0

def plotEarth():
    "Draw Earth as a globe at the origin"
    Earth_radius = 6371 # km
    global max_radius
    max_radius = max(max_radius, Earth_radius)

    # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1
    coefs = (1, 1, 1)

    # Radii corresponding to the coefficients:
    rx, ry, rz = [Earth_radius/np.sqrt(coef) for coef in coefs]

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='g')


def plotOrbit(semi_major_axis, eccentricity=0, inclination=0,
              right_ascension=0, argument_perigee=0, true_anomaly=0, label=None):
    "Draws orbit around an earth in units of kilometers."
    # Rotation matrix for inclination
    inc = inclination * pi / 180.;
    R = np.matrix([[1, 0, 0],
                   [0, cos(inc), -sin(inc)],
                   [0, sin(inc), cos(inc)]    ])

    # Rotation matrix for argument of perigee + right ascension
    rot = (right_ascension + argument_perigee) * pi/180
    R2 = np.matrix([[cos(rot), -sin(rot), 0],
                    [sin(rot), cos(rot), 0],
                    [0, 0, 1]    ])

    ### Draw orbit
    theta = np.linspace(0,2*pi, 360)
    r = (semi_major_axis * (1-eccentricity**2)) / (1 + eccentricity*cos(theta))

    xr = r*cos(theta)
    yr = r*sin(theta)
    zr = 0 * theta

    pts = np.matrix(zip(xr,yr,zr))

    # Rotate by inclination
    # Rotate by ascension + perigee
    pts =  (R * R2 * pts.T).T


    # Turn back into 1d vectors
    xr,yr,zr = pts[:,0].A.flatten(), pts[:,1].A.flatten(), pts[:,2].A.flatten()

    # Plot the orbit
    ax.plot(xr, yr, zr, '-')
    # plt.xlabel('X (km)')
    # plt.ylabel('Y (km)')
    # plt.zlabel('Z (km)')

    # Plot the satellite
    sat_angle = true_anomaly * pi/180
    satr = (semi_major_axis * (1-eccentricity**2)) / (1 + eccentricity*cos(sat_angle))
    satx = satr * cos(sat_angle)
    saty = satr * sin(sat_angle)
    satz = 0

    sat = (R * R2 * np.matrix([satx, saty, satz]).T ).flatten()
    satx = sat[0,0]
    saty = sat[0,1]
    satz = sat[0,2]

    c = np.sqrt(satx*satx + saty*saty)
    lat = np.arctan2(satz, c) * 180/pi
    lon = np.arctan2(saty, satx) * 180/pi
    print "%s : Lat: %g° Long: %g" % (label, lat, lon)

    # Draw radius vector from earth
    # ax.plot([0, satx], [0, saty], [0, satz], 'r-')
    # Draw red sphere for satellite
    ax.plot([satx],[saty],[satz], 'ro')

    global max_radius
    max_radius = max(max(r), max_radius)

    # Write satellite name next to it
    if label:
        ax.text(satx, saty, satz, label, fontsize=12)

def doDraw():
    # Adjustment of the axes, so that they all have the same span:
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    # Draw figure
plt.show()
