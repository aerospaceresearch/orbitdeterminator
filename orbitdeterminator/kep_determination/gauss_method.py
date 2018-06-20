"""Implements Gauss' method for orbit determination from three topocentric
    angular measurements of celestial bodies.
"""

import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from scipy.optimize import least_squares

# the parallax constants S and C are defined by
# S=rho cos phi' C=rho sin phi'
# rho: slant range
# phi': geocentric latitude
# We have the following:
# phi' = atan(S/C)

# compute Greenwich mean sidereal time (in hours) at UT instant of Julian date JD0:
def gmst(jd0, ut):
    return 6.656306 + 0.0657098242*(jd0-2445700.5) + 1.0027379093*ut

# compute local sidereal time from GMST and longitude EAST of Greenwich:
def localsidtime(gmst_hrs, long):
    return np.mod((gmst_hrs+long*(1.0/15.0)),24.0)

#top of page 266, chapter 5, Orbital Mechanics book
def observerpos(long, parallax_s, parallax_c, jd0, ut):

    # compute geocentric latitude from parallax constants S and C
    phi_gc = np.arctan2(parallax_s, parallax_c)
    # compute Greenwich mean sidereal time (in hours) at UT instant of JD0 date:
    gmst_hrs = gmst(jd0, ut)
    # compute local sidereal time from GMST and longitude EAST of Greenwich:
    lst_hrs = localsidtime(gmst_hrs, long)
    # Earth's mean radius in kilometers
    Re = 6378.0

    x_gc = Re*np.cos(phi_gc)*np.cos(15.0*lst_hrs)
    y_gc = Re*np.cos(phi_gc)*np.sin(15.0*lst_hrs)
    z_gc = Re*np.sin(phi_gc)

    return np.array((x_gc,y_gc,z_gc))


