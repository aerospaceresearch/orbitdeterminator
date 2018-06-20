"""Implements Gauss' method for orbit determination from three topocentric
    angular measurements of celestial bodies.
"""

import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from scipy.optimize import least_squares

def load_data_mpc(fname):
    '''
    Loads minor planet position observation data from MPC-formatted files.
    MPC format for minor planet observations is described at
    https://www.minorplanetcenter.net/iau/info/OpticalObs.html
    TODO: Add support for comets and natural satellites.
    Add support for radar observations:
    https://www.minorplanetcenter.net/iau/info/RadarObs.html
    See also NOTE 2 in:
    https://www.minorplanetcenter.net/iau/info/OpticalObs.html

    Args:
        fname (string): name of the MPC-formatted text file to be parsed

    Returns:
        x (numpy array): array of minor planet position observations following the
        MPC format.
    '''
    # dt is the dtype for MPC-formatted text files
    dt = 'i8,S7,S1,S1,S1,i8,i8,i8,f8,i8,i8,f8,i8,i8,f8,S9,S6,S6,S3'
    # mpc_names correspond to the dtype names of each field
    mpc_names = ['mpnum','provdesig','discovery','publishnote','j2000','yr','month','day','utc','ra_hr','ra_min','ra_sec','dec_deg','dec_min','dec_sec','9xblank','magband','6xblank','observatory']
    # mpc_delims are the fixed-width column delimiter following MPC format description
    mpc_delims = [5,7,1,1,1,4,3,3,7,2,3,7,3,3,6,9,6,6,3]
    return np.genfromtxt(fname, dtype=dt, names=mpc_names, delimiter=mpc_delims, autostrip=True)

# the parallax constants S and C are defined by
# S=rho cos phi' C=rho sin phi'
# rho: slant range
# phi': geocentric latitude
# We have the following:
# phi' = atan(S/C)
# rho = sqrt(S**2+C**2)

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

x = load_data_mpc('../example_data/mpc_data.txt')

# print('x[15] = ', x[15])
# print('x[\'ra_hr\'] = ', x['ra_hr'][0:10])
# print('x[\'ra_min\'] = ', x['ra_min'][0:10]/60.0)
# print('x[\'ra_sec\'] = ', x['ra_sec'][0:10]/3600.0)
print('ra  (hrs) = ', x['ra_hr'][6:18]+x['ra_min'][6:18]/60.0+x['ra_sec'][6:18]/3600.0)
print('dec (deg) = ', x['dec_deg'][6:18]+x['dec_min'][6:18]/60.0+x['dec_sec'][6:18]/3600.0)

ind_0 = 0
ind_end = 1409

ra_hrs = x['ra_hr'][ind_0:ind_end]+x['ra_min'][ind_0:ind_end]/60.0+x['ra_sec'][ind_0:ind_end]/3600.0
dec_deg = x['dec_deg'][ind_0:ind_end]+x['dec_min'][ind_0:ind_end]/60.0+x['dec_sec'][ind_0:ind_end]/3600.0

plt.plot( ra_hrs, dec_deg ) #, label='...')
plt.scatter( ra_hrs, dec_deg ) #, label='...')
plt.xlabel('ra')
plt.ylabel('dec')
plt.title('ra,dec')
# plt.legend()
plt.show()

