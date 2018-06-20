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
    return np.mod(6.656306 + 0.0657098242*(jd0-2445700.5) + 1.0027379093*ut, 24.0)

# compute Greenwich apparent sidereal time (in hours) at UT instant of Julian date JD0:
# delta_lambda: nutation in longitude (hours)
# epsilon: obliquity of the ecliptic (degrees)
def gast(jd0, ut, delta_lambda, epsilon):
    gmst_hrs = gmst(jd0, ut)
    return np.mod(gmst_hrs+delta_lambda*np.cos(epsilon), 24.0)

# compute local sidereal time from GMST and longitude EAST of Greenwich:
def localsidtime(gmst_hrs, long):
    return np.mod((gmst_hrs+long/15.0),24.0)

# geocentric observer position at a given longitude,
# parallax constants S and C, Julian date jd0 and UT time ut
# formula taken from top of page 266, chapter 5, Orbital Mechanics book
def observerpos(long, parallax_s, parallax_c, jd0, ut):

    # compute geocentric latitude from parallax constants S and C
    phi_gc = np.arctan2(parallax_s, parallax_c)
    # compute geocentric radius
    rho_gc = np.sqrt(parallax_s**2+parallax_c**2)
    # compute Greenwich mean sidereal time (in hours) at UT instant of JD0 date:
    gmst_hrs = gmst(jd0, ut)
    # compute local sidereal time from GMST and longitude EAST of Greenwich:
    lst_hrs = localsidtime(gmst_hrs, long)
    # Earth's equatorial radius in kilometers
    Re = 6378.0

    # compute cartesian components of geocentric observer position
    x_gc = Re*rho_gc*np.cos(phi_gc)*np.cos(15.0*lst_hrs)
    y_gc = Re*rho_gc*np.cos(phi_gc)*np.sin(15.0*lst_hrs)
    z_gc = Re*rho_gc*np.sin(phi_gc)

    return np.array((x_gc,y_gc,z_gc))

# TODO: implement Gauss' method, from algorithm 5.5, chapter 5, page 285, Orbital Mechanics book
# input: three pairs of topocentric (ra_i, dec_i), three geocentric observer vectors R_i,
#        and three observations times t_i, i= 1,2,3
# output: cartesian state [x0,y0,z0,u0,v0,w0] at reference epoch t0

#########################

# an example of computation of sidereal times, which will be added as unit testing:
# Determine the true and mean sidereal time on 01 January 1982 at 1h CET for
# Munich (lambda = -11deg36.5', delta_lambda = -15''.476, epsilon = 23deg26'27'')

jd0_ = 2444970.5 #Julian date of 1982, January 1st
ut = 0.0
munich_long = (11.0+36.5/60.0) #degrees
my_d_lamb = -(15.476)/15.0/3600.0 # hours
epsilon_ = np.deg2rad( 23.0+26.0/60.0+27.0/3600.0 ) #radians

print('JD = ', 2444970.5)
print('dl*cos(e) = ', my_d_lamb*np.cos(epsilon_), 's')

mu_gmst = gmst(jd0_,ut)
mu_gmst_hrs = np.floor(mu_gmst)
mu_gmst_min = np.floor((mu_gmst-mu_gmst_hrs)*60.0)
mu_gmst_sec = ((mu_gmst-mu_gmst_hrs)*60.0-mu_gmst_min)*60.0

mu_gast = gast(jd0_, ut, my_d_lamb, epsilon_)
mu_gast_hrs = np.floor(mu_gast)
mu_gast_min = np.floor((mu_gast-mu_gast_hrs)*60.0)
mu_gast_sec = ((mu_gast-mu_gast_hrs)*60.0-mu_gast_min)*60.0

mu_lmst = localsidtime(mu_gmst, munich_long)
mu_lmst_hrs = np.floor(mu_lmst)
mu_lmst_min = np.floor((mu_lmst-mu_lmst_hrs)*60.0)
mu_lmst_sec = ((mu_lmst-mu_lmst_hrs)*60.0-mu_lmst_min)*60.0

mu_last = localsidtime(mu_gast, munich_long)
mu_last_hrs = np.floor(mu_last)
mu_last_min = np.floor((mu_last-mu_last_hrs)*60.0)
mu_last_sec = ((mu_last-mu_last_hrs)*60.0-mu_last_min)*60.0

print('GMST = ', mu_gmst )
print('mu_gmst_hrs = ', mu_gmst_hrs)
print('mu_gmst_min = ', mu_gmst_min)
print('mu_gmst_sec = ', mu_gmst_sec)

print('GAST = ', mu_gast )
print('mu_gast_hrs = ', mu_gast_hrs)
print('mu_gast_min = ', mu_gast_min)
print('mu_gast_sec = ', mu_gast_sec)

print('LMST = ', mu_lmst )
print('mu_lmst_hrs = ', mu_lmst_hrs)
print('mu_lmst_min = ', mu_lmst_min)
print('mu_lmst_sec = ', mu_lmst_sec)

print('LAST = ', mu_last )
print('mu_last_hrs = ', mu_last_hrs)
print('mu_last_min = ', mu_last_min)
print('mu_last_sec = ', mu_last_sec)

# Julian date of Apophis discovery observations:
jd = 2453079.5 # 2004 Mar 15
ut = 24.0*0.10789 # UT time of 1st observation

# longitude and parallax constants C,S for observatory with code 691:
# 248.4010  0.84951  +0.52642
# taken from https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html
# retrieved on: 19 Jun 2018

long_691 = 248.4010 # degrees
C_691 = 0.84951
S_691 = +0.52642

#geocentric observer position at time of 1st Apophis observation:
pos_691 = observerpos(long_691, S_691, C_691, jd, ut)
print('pos_691 = ', pos_691)

# cross-check:
radius_ = np.sqrt(pos_691[0]**2+pos_691[1]**2+pos_691[2]**2)
print('radius_ = ', radius_)


