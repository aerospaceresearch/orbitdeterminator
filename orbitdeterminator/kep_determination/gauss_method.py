# TODO: evaluate Earth ephemeris only once for a given TDB instant
# this implies saving all UTC times and their TDB equivalencies

"""Implements Gauss' method for orbit determination from three topocentric
    angular measurements of celestial bodies.
"""

import math
import numpy as np
from astropy.coordinates import Longitude, Angle, SkyCoord
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
from astropy import units as uts
from astropy import constants as cts
from astropy.time import Time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from poliastro.stumpff import c2, c3
from least_squares import xyz_frame_, orbel2xyz, truean2eccan, meanmotion
from astropy.coordinates.earth_orientation import obliquity
from astropy.coordinates.matrix_utilities import rotation_matrix
from scipy.optimize import least_squares

au = cts.au.to(uts.Unit('km')).value
mu_Sun = cts.GM_sun.to(uts.Unit('au3 / day2')).value
mu_Earth = cts.GM_earth.to(uts.Unit('km3 / s2')).value
c_light = cts.c.to(uts.Unit('au/day'))
earth_f = 0.003353
Re = cts.R_earth.to(uts.Unit('km')).value

# load JPL DE432s ephemeris SPK kernel
# 'de432s.bsp' is automatically loaded by astropy, via jplephem
# 'de432s.bsp' is about 10MB in size and will be automatically downloaded if not present yet in astropy's cache
# for more information, see astropy.coordinates.solar_system_ephemeris documentation
solar_system_ephemeris.set('de432s')

obliquity_j2000 = obliquity(2451544.5) # mean obliquity of the ecliptic at J2000.0
rot_equat_to_eclip = rotation_matrix( obliquity_j2000, 'x') #rotation matrix from equatorial to ecliptic frames
rot_eclip_to_equat = rotation_matrix(-obliquity_j2000, 'x') #rotation matrix from ecliptic to equatorial frames

# print('obliquity_j2000 = ', obliquity_j2000)
# print('rot_equat_to_eclip = ', rot_equat_to_eclip)

np.set_printoptions(precision=16)

def load_mpc_observatories_data(mpc_observatories_fname):
    obs_dt = 'S3, f8, f8, f8, S48'
    obs_delims = [4, 10, 9, 10, 48]
    return np.genfromtxt(mpc_observatories_fname, dtype=obs_dt, names=True, delimiter=obs_delims, autostrip=True, encoding=None)

def load_sat_observatories_data(sat_observatories_fname):
    obs_dt = 'i8, S2, f8, f8, f8, S18'
    obs_delims = [4, 3, 10, 10, 8, 21]
    return np.genfromtxt(sat_observatories_fname, dtype=obs_dt, names=True, delimiter=obs_delims, autostrip=True, encoding=None, skip_header=1)

def get_observatory_data(observatory_code, mpc_observatories_data):
    # print('observatory_code = ', observatory_code)
    # print('mpc_observatories_data[\'Code\'] = ', mpc_observatories_data['Code'])
    arr_index = np.where(mpc_observatories_data['Code'] == observatory_code)
    # print('arr_index = ', arr_index)
    # print('mpc_observatories_data[arr_index] = ', mpc_observatories_data[arr_index])
    # print('arr_index = ', arr_index)
    # print('arr_index[0] = ', arr_index[0])
    # print('arr_index[0][0] = ', arr_index[0][0])
    # print('mpc_observatories_data[arr_index[0]] = ', mpc_observatories_data[arr_index[0]])
    # print('mpc_observatories_data[arr_index[0][0]] = ', mpc_observatories_data[arr_index[0][0]])
    return mpc_observatories_data[arr_index[0][0]]

def get_station_data(station_code, sat_observatories_data):
    # print('station_code = ', station_code)
    # print('sat_observatories_data[\'No\'] = ', sat_observatories_data['No'])
    arr_index = np.where(sat_observatories_data['No'] == station_code)
    # print('arr_index = ', arr_index)
    # print('sat_observatories_data[arr_index] = ', sat_observatories_data[arr_index])
    # print('arr_index[0] = ', arr_index[0])
    # print('arr_index[0][0] = ', arr_index[0][0])
    # print('sat_observatories_data[arr_index[0]] = ', sat_observatories_data[arr_index[0]])
    # print('sat_observatories_data[arr_index[0][0]] = ', sat_observatories_data[arr_index[0][0]])
    return sat_observatories_data[arr_index[0][0]]

def load_mpc_data(fname):
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
    dt = 'i8,S7,S1,S1,S1,i8,i8,i8,f8,U24,S9,S6,S6,S3'
    # mpc_names correspond to the dtype names of each field
    mpc_names = ['mpnum','provdesig','discovery','publishnote','j2000','yr','month','day','utc','radec','9xblank','magband','6xblank','observatory']
    # mpc_delims are the fixed-width column delimiter following MPC format description
    mpc_delims = [5,7,1,1,1,4,3,3,7,24,9,6,6,3]
    return np.genfromtxt(fname, dtype=dt, names=mpc_names, delimiter=mpc_delims, autostrip=True)

def load_iod_data(fname):
    '''
    Loads satellite position observation data files following the Interactive
    Orbit Determination format (IOD). Currently, the only supported angle format
    is 2, as specified in IOD format. IOD format is described at
    http://www.satobs.org/position/IODformat.html. IOD sub-format 2 corresponds to:

              50        60
             89 123456789 1234
    RA/DEC = HHMMmmm+DDMMmm MX   (MX in minutes of arc)

    TODO: add other IOD angle sub-formats; construct numpy array according to different
    angle formats.

    Args:
        fname (string): name of the IOD-formatted text file to be parsed

    Returns:
        x (numpy array): array of satellite position observations following the
        IOD format, with angle format code = 2.
    '''
    # dt is the dtype for IOD-formatted text files
    dt = 'S15, i8, S1, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, S1, S1'
    # iod_names correspond to the dtype names of each field
    iod_names = ['object', 'station', 'stationstatus', 'yr', 'month', 'day',
        'hr', 'min', 'sec', 'msec', 'timeM', 'timeX', 'angformat', 'epoch', 'raHH',
        'raMM','rammm','decDD','decMM','decmmm','radecM','radecX','optical','vismag']
    # TODO: read first line, get sub-format, construct iod_delims from there
    # as it is now, it only works for the given test file iod_data.txt
    # iod_delims are the fixed-width column delimiter followinf IOD format description
    iod_delims = [15, 5, 2, 5, 2, 2,
        2, 2, 2, 3, 2, 1, 2, 1, 3,
        2, 3, 3, 2, 2, 2, 1, 2, 1]
    return np.genfromtxt(fname, dtype=dt, names=iod_names, delimiter=iod_delims, autostrip=True)

# Compute geocentric observer position at UTC instant t_utc, for Sun-centered orbits,
# at a given observation site defined by its longitude, and parallax constants S and C
# formula taken from top of page 266, chapter 5, Orbital Mechanics book (Curtis)
# the parallax constants S and C are defined by
# S=rho cos phi' C=rho sin phi'
# rho: slant range
# phi': geocentric latitude
# We have the following:
# phi' = atan(S/C)
# rho = sqrt(S**2+C**2)
def observerpos_mpc(long, parallax_s, parallax_c, t_utc):

    # Earth's equatorial radius in kilometers
    # Re = cts.R_earth.to(uts.Unit('km')).value

    # define Longitude object for the observation site longitude
    long_site = Longitude(long, uts.degree, wrap_angle=360.0*uts.degree)
    # print('long_site = ', long_site)
    # compute sidereal time of observation at site
    t_site_lmst = t_utc.sidereal_time('mean', longitude=long_site)
    lmst_rad = t_site_lmst.rad # np.deg2rad(lmst_hrs*15.0) # radians

    # compute cartesian components of geocentric (non rotating) observer position
    x_gc = Re*parallax_c*np.cos(lmst_rad)
    y_gc = Re*parallax_c*np.sin(lmst_rad)
    z_gc = Re*parallax_s

    # print('x_gc = ', x_gc)
    # print('y_gc = ', y_gc)
    # print('z_gc = ', z_gc)

    return np.array((x_gc,y_gc,z_gc))

# Compute geocentric observer position at UTC instant t_utc, for Earth-centered orbits,
# at a given observation site defined by its longitude, latitude and elevation above reference ellipsoid
# formula taken from bottom of page 265 (Eq. 5.56), chapter 5, Orbital Mechanics book (Curtis)
# lat: geodetic latitude (deg)
# long: longitude (deg)
# elev: elevation above reference ellipsoid (m)
# t_utc: time (UTC)
def observerpos_sat(lat, long, elev, t_utc):

    # Earth's equatorial radius in kilometers
    # Re = cts.R_earth.to(uts.Unit('km')).value

    # define Longitude object for the observation site longitude
    # print('long = ', long)
    # print('lat = ', lat)
    long_site = Longitude(long, uts.degree, wrap_angle=180.0*uts.degree)
    # print('long_site = ', long_site)
    # long_site = Longitude(long, uts.degree, wrap_angle=360.0*uts.degree)
    # print('long_site = ', long_site)
    # compute sidereal time of observation at site
    t_site_lmst = t_utc.sidereal_time('mean', longitude=long_site)
    # print('t_site_lmst = ', t_site_lmst)
    # print('t_lmgt = ', t_utc.sidereal_time('mean', 'greenwich'))
    lmst_rad = t_site_lmst.rad # np.deg2rad(lmst_hrs*15.0) # radians

    # latitude
    phi_rad = np.deg2rad(lat)

    # convert ellipsoid coordinates to cartesian
    cos_phi = np.cos( phi_rad )
    cos_phi_cos_theta = cos_phi*np.cos( lmst_rad )
    cos_phi_sin_theta = cos_phi*np.sin( lmst_rad )
    sin_phi = np.sin( phi_rad )
    denum = np.sqrt(1.0-(2.0*earth_f-earth_f**2)*sin_phi**2)
    r_xy = Re/denum+elev/1000.0
    r_z = Re*((1.0-earth_f)**2)/denum+elev/1000.0

    # compute cartesian components of geocentric (non-rotating) observer position
    x_gc = r_xy*cos_phi_cos_theta
    y_gc = r_xy*cos_phi_sin_theta
    z_gc = r_z*sin_phi

    # print('x_gc = ', x_gc)
    # print('y_gc = ', y_gc)
    # print('z_gc = ', z_gc)

    return np.array((x_gc,y_gc,z_gc))

#observerpos_sat_book: compute the geocentric position of observer (Earth-centered orbit)
#this function works to run examples from Orbital Mechanics book
# formula taken from bottom of page 265 (Eq. 5.56), chapter 5, Orbital Mechanics book (Curtis)
#phi_deg: geodetic latitude (phi), degrees
#altitude_km: altitude above reference ellipsoid, kilometers
#f: Earth's flattening/oblateness factor (adimensional)
#lst_deg: local sidereal time, degrees
def observerpos_sat_book(phi_deg, altitude_km, f, lst_deg):
    # Earth's equatorial radius in kilometers
    # Re = cts.R_earth.to(uts.Unit('km')).value
    phi_rad = np.deg2rad(phi_deg)
    cos_phi = np.cos( phi_rad )
    lst_rad = np.deg2rad(lst_deg)
    cos_phi_cos_theta = cos_phi*np.cos( lst_rad )
    cos_phi_sin_theta = cos_phi*np.sin( lst_rad )
    sin_phi = np.sin( phi_rad )
    denum = np.sqrt(1.0-(2.0*f-f**2)*sin_phi**2)
    r_xy = Re/denum+altitude_km
    r_z = Re*((1-f)**2)/denum+altitude_km

    # compute cartesian components of geocentric observer position
    x_gc = r_xy*cos_phi_cos_theta
    y_gc = r_xy*cos_phi_sin_theta
    z_gc = r_z*sin_phi

    return np.array((x_gc,y_gc,z_gc))

#Line-Of-Sight (LOS) vector
#ra must be in rad, dec must be in rad
def losvector(ra_rad, dec_rad):
    cosa_cosd = np.cos(ra_rad)*np.cos(dec_rad)
    sina_cosd = np.sin(ra_rad)*np.cos(dec_rad)
    sind = np.sin(dec_rad)
    return np.array((cosa_cosd, sina_cosd, sind))

def lagrangef(mu, r2, tau):
    return 1.0-0.5*(mu/(r2**3))*(tau**2)

def lagrangeg(mu, r2, tau):
    return tau-(1.0/6.0)*(mu/(r2**3))*(tau**3)

# Set of functions for cartesian states -> Keplerian elements

def kep_h_norm(x, y, z, u, v, w):
    return np.sqrt( (y*w-z*v)**2 + (z*u-x*w)**2 + (x*v-y*u)**2 )

def kep_h_vec(x, y, z, u, v, w):
    return np.array((y*w-z*v, z*u-x*w, x*v-y*u))

def semimajoraxis(x, y, z, u, v, w, mu):
    myRadius=np.sqrt((x**2)+(y**2)+(z**2))
    myVelSqr=(u**2)+(v**2)+(w**2)
    return 1.0/( (2.0/myRadius)-(myVelSqr/mu) )

def eccentricity(x, y, z, u, v, w, mu):
    h2 = ((y*w-z*v)**2) + ((z*u-x*w)**2) + ((x*v-y*u)**2)
    a = semimajoraxis(x,y,z,u,v,w,mu)
    quotient = h2/( mu*a )
    return np.sqrt(1.0 - quotient)

def inclination(x, y, z, u, v, w):
    my_hz = x*v-y*u
    my_h = np.sqrt( (y*w-z*v)**2 + (z*u-x*w)**2 + (x*v-y*u)**2 )

    return np.arccos(my_hz/my_h)

def longascnode(x, y, z, u, v, w):
    #the longitude of ascending node is computed as
    #the angle between x-axis and the vector n = (-hy,hx,0)
    #where hx, hy, are resp., the x and y comps. of ang mom vector per unit mass, h
    res = np.arctan2(y*w-z*v, x*w-z*u) # remember atan2 is atan2(y/x)
    if res >= 0.0:
        return res
    else:
        return res+2.0*np.pi

def rungelenz(x, y, z, u, v, w, mu):
    r = np.sqrt(x**2+y**2+z**2)
    lrl_x = ( -(z*u-x*w)*w+(x*v-y*u)*v )/mu-x/r
    lrl_y = ( -(x*v-y*u)*u+(y*w-z*v)*w )/mu-y/r
    lrl_z = ( -(y*w-z*v)*v+(z*u-x*w)*u )/mu-z/r
    return np.array((lrl_x, lrl_y, lrl_z))

def argperi(x, y, z, u, v, w, mu):
    #n = (z-axis unit vector)×h = (-hy, hx, 0)
    n = np.array((x*w-z*u, y*w-z*v, 0.0))
    e = rungelenz(x,y,z,u,v,w,mu) #cartesian comps. of Laplace-Runge-Lenz vector
    n = n/np.sqrt(n[0]**2+n[1]**2+n[2]**2)
    e = e/np.sqrt(e[0]**2+e[1]**2+e[2]**2)
    cos_omega = np.dot(n, e)

    if e[2] >= 0.0:
        return np.arccos(cos_omega)
    else:
        return 2.0*np.pi-np.arccos(cos_omega)

def trueanomaly(x, y, z, u, v, w, mu):
    r_vec = np.array((x, y, z))
    r_vec = r_vec/np.linalg.norm(r_vec, ord=2)
    e_vec = rungelenz(x, y, z, u, v, w, mu)
    e_vec = e_vec/np.linalg.norm(e_vec, ord=2)
    v_vec = np.array((u, v, w))
    v_r_num = np.dot(v_vec, r_vec)
    if v_r_num>=0.0:
        return np.arccos(np.dot(e_vec, r_vec))
    elif v_r_num<0.0:
        return 2.0*np.pi-np.arccos(np.dot(e_vec, r_vec))

def taupericenter(t, e, f, n):
    E0 = truean2eccan(e, f)
    M0 = np.mod(E0-e*np.sin(E0), 2.0*np.pi)
    return t-M0/n

# alpha = 1/a
def alpha(x, y, z, u, v, w, mu):
    myRadius=np.sqrt((x**2)+(y**2)+(z**2))
    myVelSqr=(u**2)+(v**2)+(w**2)
    return (2.0/myRadius)-(myVelSqr/mu)

#vr0 is the radial velocity $vr0 = \vec r_0 \cdot \vec r_0$
def univkepler(dt, x, y, z, u, v, w, mu, iters=5, atol=1e-15):
    # compute preliminaries
    r0 = np.sqrt((x**2)+(y**2)+(z**2))
    v20 = (u**2)+(v**2)+(w**2)
    vr0 = (x*u+y*v+z*w)/r0
    alpha0 = (2.0/r0)-(v20/mu)
    # compute initial estimate for xi
    xi = np.sqrt(mu)*np.abs(alpha0)*dt
    i = 0
    ratio_i = 1.0
    while np.abs(ratio_i)>atol and i<iters:
        # print('* i = ', i)
        # print(' * xi = ', xi)
        # print(' * alpha0 = ', alpha0)
        xi2 = xi**2
        z_i = alpha0*(xi2)
        a_i = (r0*vr0)/np.sqrt(mu)
        b_i = 1.0-alpha0*r0
        # print(' * z_i = ', z_i)
        C_z_i = c2(z_i)
        S_z_i = c3(z_i)
        f_i = a_i*xi2*C_z_i + b_i*(xi**3)*S_z_i + r0*xi - np.sqrt(mu)*dt
        g_i = a_i*xi*(1.0-z_i*S_z_i) + b_i*xi2*C_z_i+r0
        ratio_i = f_i/g_i
        xi = xi - ratio_i
        i += 1
        # print('i = ', i, ', ratio_i = ', ratio_i)
    return xi

def lagrangef_(xi, z, r):
    # print('xi = ', xi)
    # print('z = ', z)
    # print('r = ', r)
    return 1.0-(xi**2)*c2(z)/r

def lagrangeg_(tau, xi, z, mu):
    return tau-(xi**3)*c3(z)/np.sqrt(mu)

def get_observations_data(mpc_object_data, inds):
    # construct SkyCoord 3-element array with observational information
    timeobs = np.zeros((3,), dtype=Time)
    obs_radec = np.zeros((3,), dtype=SkyCoord)
    obs_t = np.zeros((3,))

    timeobs[0] = Time( datetime(mpc_object_data['yr'][inds[0]], mpc_object_data['month'][inds[0]], mpc_object_data['day'][inds[0]]) + timedelta(days=mpc_object_data['utc'][inds[0]]) )
    timeobs[1] = Time( datetime(mpc_object_data['yr'][inds[1]], mpc_object_data['month'][inds[1]], mpc_object_data['day'][inds[1]]) + timedelta(days=mpc_object_data['utc'][inds[1]]) )
    timeobs[2] = Time( datetime(mpc_object_data['yr'][inds[2]], mpc_object_data['month'][inds[2]], mpc_object_data['day'][inds[2]]) + timedelta(days=mpc_object_data['utc'][inds[2]]) )

    obs_radec[0] = SkyCoord(mpc_object_data['radec'][inds[0]], unit=(uts.hourangle, uts.deg), obstime=timeobs[0])
    obs_radec[1] = SkyCoord(mpc_object_data['radec'][inds[1]], unit=(uts.hourangle, uts.deg), obstime=timeobs[1])
    obs_radec[2] = SkyCoord(mpc_object_data['radec'][inds[2]], unit=(uts.hourangle, uts.deg), obstime=timeobs[2])

    # print('obs_radec[0] = ', obs_radec[0])
    # print('obs_radec[1] = ', obs_radec[1])
    # print('obs_radec[2] = ', obs_radec[2])

    # construct vector of observation time (continous variable)
    obs_t[0] = obs_radec[0].obstime.tdb.jd
    obs_t[1] = obs_radec[1].obstime.tdb.jd
    obs_t[2] = obs_radec[2].obstime.tdb.jd

    site_codes = [mpc_object_data['observatory'][inds[0]], mpc_object_data['observatory'][inds[1]], mpc_object_data['observatory'][inds[2]]]

    return obs_radec, obs_t, site_codes

def get_observations_data_sat(iod_object_data, inds):
    # construct SkyCoord 3-element array with observational information
    timeobs = np.zeros((3,), dtype=Time)
    obs_radec = np.zeros((3,), dtype=SkyCoord)
    obs_t = np.zeros((3,))

    # print('iod_object_data[\'hr\'][inds[0]] = ', iod_object_data['hr'][inds[0]])
    # print('iod_object_data[\'min\'][inds[0]] = ', iod_object_data['min'][inds[0]])
    # print('iod_object_data[\'sec\'][inds[0]] = ', iod_object_data['sec'][inds[0]])
    # print('iod_object_data[\'msec\'][inds[0]] = ', iod_object_data['msec'][inds[0]])

    td1 = timedelta(hours=1.0*iod_object_data['hr'][inds[0]], minutes=1.0*iod_object_data['min'][inds[0]], seconds=(iod_object_data['sec'][inds[0]]+iod_object_data['msec'][inds[0]]/1000.0))
    td2 = timedelta(hours=1.0*iod_object_data['hr'][inds[1]], minutes=1.0*iod_object_data['min'][inds[1]], seconds=(iod_object_data['sec'][inds[1]]+iod_object_data['msec'][inds[1]]/1000.0))
    td3 = timedelta(hours=1.0*iod_object_data['hr'][inds[2]], minutes=1.0*iod_object_data['min'][inds[2]], seconds=(iod_object_data['sec'][inds[2]]+iod_object_data['msec'][inds[2]]/1000.0))

    timeobs[0] = Time( datetime(iod_object_data['yr'][inds[0]], iod_object_data['month'][inds[0]], iod_object_data['day'][inds[0]]) + td1 )
    timeobs[1] = Time( datetime(iod_object_data['yr'][inds[1]], iod_object_data['month'][inds[1]], iod_object_data['day'][inds[1]]) + td2 )
    timeobs[2] = Time( datetime(iod_object_data['yr'][inds[2]], iod_object_data['month'][inds[2]], iod_object_data['day'][inds[2]]) + td3 )

    # print('timeobs = ', timeobs)
    # print('timeobs00 = ', (timeobs[0]-timeobs[0]).sec)
    # print('timeobs10 = ', (timeobs[1]-timeobs[0]).sec)
    # print('timeobs20 = ', (timeobs[2]-timeobs[0]).sec)

    raHHMMmmm0 = iod_object_data['raHH'][inds[0]] + (iod_object_data['raMM'][inds[0]]+iod_object_data['rammm'][inds[0]]/1000.0)/60.0
    raHHMMmmm1 = iod_object_data['raHH'][inds[1]] + (iod_object_data['raMM'][inds[1]]+iod_object_data['rammm'][inds[1]]/1000.0)/60.0
    raHHMMmmm2 = iod_object_data['raHH'][inds[2]] + (iod_object_data['raMM'][inds[2]]+iod_object_data['rammm'][inds[2]]/1000.0)/60.0

    decDDMMmmm0 = iod_object_data['decDD'][inds[0]] + (iod_object_data['decMM'][inds[0]]+iod_object_data['decmmm'][inds[0]]/1000.0)/60.0
    decDDMMmmm1 = iod_object_data['decDD'][inds[1]] + (iod_object_data['decMM'][inds[1]]+iod_object_data['decmmm'][inds[1]]/1000.0)/60.0
    decDDMMmmm2 = iod_object_data['decDD'][inds[2]] + (iod_object_data['decMM'][inds[2]]+iod_object_data['decmmm'][inds[2]]/1000.0)/60.0

    obs_radec[0] = SkyCoord(ra=raHHMMmmm0, dec=decDDMMmmm0, unit=(uts.hourangle, uts.deg), obstime=timeobs[0])
    obs_radec[1] = SkyCoord(ra=raHHMMmmm1, dec=decDDMMmmm1, unit=(uts.hourangle, uts.deg), obstime=timeobs[1])
    obs_radec[2] = SkyCoord(ra=raHHMMmmm2, dec=decDDMMmmm2, unit=(uts.hourangle, uts.deg), obstime=timeobs[2])

    # print('obs_radec[0] = ', obs_radec[0])
    # print('obs_radec[1] = ', obs_radec[1])
    # print('obs_radec[2] = ', obs_radec[2])

    # construct vector of observation time (continous variable)
    obs_t[0] = (timeobs[0]-timeobs[0]).sec
    obs_t[1] = (timeobs[1]-timeobs[0]).sec
    obs_t[2] = (timeobs[2]-timeobs[0]).sec

    site_codes = [iod_object_data['station'][inds[0]], iod_object_data['station'][inds[1]], iod_object_data['station'][inds[2]]]

    return obs_radec, obs_t, site_codes

# heliocentric position of Earth at Julian date t_tdb (TDB, days), according to SPK kernel defined by solar_system_ephemeris
# returns: cartesian position in km
def earth_ephemeris(t_tdb):
    t = Time(t_tdb, format='jd', scale='tdb')
    # print('t_tdb = ', t_tdb)
    # print('t = ', t)
    ye = get_body_barycentric('earth', t)
    ys = get_body_barycentric('sun', t)
    y = ye - ys
    # print('y = ', y)
    return y.xyz.value

def observer_wrt_sun(long, parallax_s, parallax_c, t_utc):
    t_jd_tdb = t_utc.tdb.jd
    xyz_es = earth_ephemeris(t_jd_tdb)
    xyz_oe = observerpos_mpc(long, parallax_s, parallax_c, t_utc)
    # print('xyz_es = ', xyz_es)
    # print('xyz_oe = ', xyz_oe)
    # print('(xyz_oe+xyz_es)/au = ', (xyz_oe+xyz_es)/au)
    return (xyz_oe+xyz_es)/au

def object_wrt_sun(t_utc, a, e, taup, omega, I, Omega):
    t_jd_tdb = t_utc.tdb.jd
    xyz_eclip = orbel2xyz(t_jd_tdb, mu_Sun, a, e, taup, omega, I, Omega)
    # print('xyz_eclip = ', xyz_eclip)
    # print('np.matmul(rot_eclip_to_equat, xyz_eclip) = ', np.matmul(rot_eclip_to_equat, xyz_eclip))
    return np.matmul(rot_eclip_to_equat, xyz_eclip)

def rho_vec(long, parallax_s, parallax_c, t_utc, a, e, taup, omega, I, Omega):
    return object_wrt_sun(t_utc, a, e, taup, omega, I, Omega)-observer_wrt_sun(long, parallax_s, parallax_c, t_utc)

def rhovec2radec(long, parallax_s, parallax_c, t_utc, a, e, taup, omega, I, Omega):
    r_v = rho_vec(long, parallax_s, parallax_c, t_utc, a, e, taup, omega, I, Omega)
    # print('r_v = ', r_v)
    r_v_norm = np.linalg.norm(r_v, ord=2)
    # print('r_v_norm = ', r_v_norm)
    # r_v = rho_vec(long, parallax_s, parallax_c, t_utc-r_v_norm/c_light, a, e, taup, omega, I, Omega)
    # r_v_norm = np.linalg.norm(r_v, ord=2)
    r_v_unit = r_v/r_v_norm
    # print('r_v_unit = ', r_v_unit)
    cosd_cosa = r_v_unit[0]
    cosd_sina = r_v_unit[1]
    sind = r_v_unit[2]
    ra_rad = np.arctan2(cosd_sina, cosd_cosa)
    dec_rad = np.arcsin(sind)
    if ra_rad <0.0:
        return ra_rad+2.0*np.pi, dec_rad
    else:
        return ra_rad, dec_rad

# signed difference between two angles
# code adapted from https://rosettacode.org/wiki/Angle_difference_between_two_bearings#Python
def angle_diff_rad(a1_rad, a2_rad):
    r = (a2_rad - a1_rad) % (2.0*np.pi)
    # Python modulus has same sign as divisor, which is positive here,
    # so no need to consider negative case
    if r >= np.pi:
        r -= (2.0*np.pi)
    return r

def radec_residual(x, t_ra_dec_datapoint, long, parallax_s, parallax_c):
    ra_comp, dec_comp = rhovec2radec(long, parallax_s, parallax_c, t_ra_dec_datapoint.obstime, x[0], x[1], x[2], x[3], x[4], x[5])
    # los_comp = losvector(ra_comp, dec_comp)
    # print('los_comp = ', los_comp)
    ra_obs, dec_obs = t_ra_dec_datapoint.ra.rad, t_ra_dec_datapoint.dec.rad
    # los_obs = losvector(ra_obs, dec_obs)
    # print('los_obs = ', los_obs)
    # print('ra_obs = ', ra_obs)
    # print('dec_obs = ', dec_obs)
    # print('ra_comp = ', ra_comp)
    # print('dec_comp = ', dec_comp)
    #"unsigned" distance between points in torus
    diff_ra = angle_diff_rad(ra_obs, ra_comp)
    diff_dec = angle_diff_rad(dec_obs, dec_comp)
    # print('diff_ra = ', np.rad2deg(diff_ra), 'deg')
    # print('diff_dec = ', np.rad2deg(diff_dec), 'deg')
    return np.array((diff_ra,diff_dec))

def radec_residual_rov(x, t, ra_obs_rad, dec_obs_rad, long, parallax_s, parallax_c):
    ra_comp, dec_comp = rhovec2radec(long, parallax_s, parallax_c, t, x[0], x[1], x[2], x[3], x[4], x[5])
    # los_comp = losvector(ra_comp, dec_comp)
    # print('los_comp = ', los_comp)
    # ra_obs, dec_obs = t_ra_dec_datapoint.ra.rad, t_ra_dec_datapoint.dec.rad
    # los_obs = losvector(ra_obs, dec_obs)
    # print('los_obs = ', los_obs)
    # print('ra_obs = ', ra_obs)
    # print('dec_obs = ', dec_obs)
    # print('ra_comp = ', ra_comp)
    # print('dec_comp = ', dec_comp)
    #"unsigned" distance between points in torus
    diff_ra = angle_diff_rad(ra_obs_rad, ra_comp)
    diff_dec = angle_diff_rad(dec_obs_rad, dec_comp)
    # print('diff_ra = ', np.rad2deg(diff_ra), 'deg')
    # print('diff_dec = ', np.rad2deg(diff_dec), 'deg')
    return np.array((diff_ra,diff_dec))

def get_observer_pos_wrt_sun(mpc_observatories_data, obs_radec, site_codes):
    # astronomical unit in km
    Ea_hc_pos = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))
    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))
    # load MPC observatory data
    obsite1 = get_observatory_data(site_codes[0], mpc_observatories_data)
    obsite2 = get_observatory_data(site_codes[1], mpc_observatories_data)
    obsite3 = get_observatory_data(site_codes[2], mpc_observatories_data)
    # print('obsite1 = ', obsite1)
    # print('obsite2 = ', obsite2)
    # print('obsite3 = ', obsite3)
    # compute TDB instant of each observation
    t_jd1_tdb_val = obs_radec[0].obstime.tdb.jd
    t_jd2_tdb_val = obs_radec[1].obstime.tdb.jd
    t_jd3_tdb_val = obs_radec[2].obstime.tdb.jd

    # print(' jd1 (tdb) = ', t_jd1_tdb_val)
    # print(' jd2 (tdb) = ', t_jd2_tdb_val)
    # print(' jd3 (tdb) = ', t_jd3_tdb_val)

    Ea_jd1 = earth_ephemeris(t_jd1_tdb_val)
    Ea_jd2 = earth_ephemeris(t_jd2_tdb_val)
    Ea_jd3 = earth_ephemeris(t_jd3_tdb_val)

    Ea_hc_pos[0] = Ea_jd1/au
    Ea_hc_pos[1] = Ea_jd2/au
    Ea_hc_pos[2] = Ea_jd3/au

    # print('Ea_hc_pos[0] = ', Ea_hc_pos[0])
    # print('Ea_hc_pos[1] = ', Ea_hc_pos[1])
    # print('Ea_hc_pos[2] = ', Ea_hc_pos[2])

    R[0] = (  Ea_jd1 + observerpos_mpc(obsite1['Long'], obsite1['sin'], obsite1['cos'], obs_radec[0].obstime)  )/au
    R[1] = (  Ea_jd2 + observerpos_mpc(obsite2['Long'], obsite2['sin'], obsite2['cos'], obs_radec[1].obstime)  )/au
    R[2] = (  Ea_jd3 + observerpos_mpc(obsite3['Long'], obsite3['sin'], obsite3['cos'], obs_radec[2].obstime)  )/au

    # print('R[0] = ', R[0])
    # print('R[1] = ', R[1])
    # print('R[2] = ', R[2])

    return R, Ea_hc_pos

def get_observer_pos_wrt_earth(sat_observatories_data, obs_radec, site_codes):
    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))
    # load MPC observatory data
    obsite1 = get_station_data(site_codes[0], sat_observatories_data)
    obsite2 = get_station_data(site_codes[1], sat_observatories_data)
    obsite3 = get_station_data(site_codes[2], sat_observatories_data)
    # print('obsite1 = ', obsite1)
    # print('obsite2 = ', obsite2)
    # print('obsite3 = ', obsite3)

    # print('obsite1[\'Latitude\'] = ', obsite1['Latitude'])
    # print('obsite2[\'Latitude\'] = ', obsite2['Latitude'])
    # print('obsite3[\'Latitude\'] = ', obsite3['Latitude'])

    # print('obsite1[\'Longitude\'] = ', obsite1['Longitude'])
    # print('obsite2[\'Longitude\'] = ', obsite2['Longitude'])
    # print('obsite3[\'Longitude\'] = ', obsite3['Longitude'])

    # print('obsite1[\'Elev\'] = ', obsite1['Elev'])
    # print('obsite2[\'Elev\'] = ', obsite2['Elev'])
    # print('obsite3[\'Elev\'] = ', obsite3['Elev'])

    # print('obs_radec[0].obstime = ', obs_radec[0].obstime)
    # print('obs_radec[1].obstime = ', obs_radec[1].obstime)
    # print('obs_radec[2].obstime = ', obs_radec[2].obstime)

    R[0] = observerpos_sat(obsite1['Latitude'], obsite1['Longitude'], obsite1['Elev'], obs_radec[0].obstime)
    R[1] = observerpos_sat(obsite2['Latitude'], obsite2['Longitude'], obsite2['Elev'], obs_radec[1].obstime)
    R[2] = observerpos_sat(obsite3['Latitude'], obsite3['Longitude'], obsite3['Elev'], obs_radec[2].obstime)

    # print('R[0] = ', R[0])
    # print('R[1] = ', R[1])
    # print('R[2] = ', R[2])

    return R

def gauss_method_core(obs_radec, obs_t, R, mu, r2_root_ind=0):
    # get Julian date of observations
    t1 = obs_t[0]
    t2 = obs_t[1]
    t3 = obs_t[2]

    # print('t1 = ', t1)
    # print('t2 = ', t2)
    # print('t3 = ', t3)

    # compute Line-Of-Sight (LOS) vectors
    rho1 = losvector(obs_radec[0].ra.rad, obs_radec[0].dec.rad)
    rho2 = losvector(obs_radec[1].ra.rad, obs_radec[1].dec.rad)
    rho3 = losvector(obs_radec[2].ra.rad, obs_radec[2].dec.rad)

    # print('rho1 = ', rho1)
    # print('rho2 = ', rho2)
    # print('rho3 = ', rho3)

    # compute time differences; make sure time units are consistent!
    tau1 = (t1-t2)
    tau3 = (t3-t2)
    tau = (tau3-tau1)

    # print('tau1 = ', tau1)
    # print('tau3 = ', tau3)
    # print('tau = ', tau)

    p = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    p[0] = np.cross(rho2, rho3)
    p[1] = np.cross(rho1, rho3)
    p[2] = np.cross(rho1, rho2)

    #print('p = ', p)

    D0  = np.dot(rho1, p[0])

    # print('D0 = ', D0)

    D = np.zeros((3,3))

    for i in range(0,3):
        for j in range(0,3):
            # print('i,j=', i, j)
            D[i,j] = np.dot(R[i], p[j])

    # print('D = ', D)

    A = (-D[0,1]*(tau3/tau)+D[1,1]+D[2,1]*(tau1/tau))/D0
    B = (D[0,1]*(tau3**2-tau**2)*(tau3/tau)+D[2,1]*(tau**2-tau1**2)*(tau1/tau))/(6*D0)

    # print('A = ', A)
    # print('B = ', B)

    E = np.dot(R[1], rho2)
    Rsub2p2 = np.dot(R[1], R[1])

    # print('E = ', E)
    # print('Rsub2p2 = ', Rsub2p2)

    a = -(A**2+2.0*A*E+Rsub2p2)
    b = -2.0*mu*B*(A+E)
    c = -(mu**2)*(B**2)

    # print('a = ', a)
    # print('b = ', b)
    # print('c = ', c)

    #get all real, positive solutions to the Gauss polynomial
    gauss_poly_coeffs = np.zeros((9,))
    gauss_poly_coeffs[0] = 1.0
    gauss_poly_coeffs[2] = a
    gauss_poly_coeffs[5] = b
    gauss_poly_coeffs[8] = c
    # 1 2 3 4 5 6 7 8 9
    # 0 1 2 3 4 5 6 7 8
    # 8 7 6 5 4 3 2 1 0
    gauss_poly_roots = np.roots(gauss_poly_coeffs)
    rt_indx = np.where( np.isreal(gauss_poly_roots) & (gauss_poly_roots >= 0.0) )
    # print('rt_indx[0] = ', rt_indx[0])
    # print('np.real(gauss_poly_roots[rt_indx])[0] = ', np.real(gauss_poly_roots[rt_indx])[0])
    if len(rt_indx[0]) > 1: #-1:#
        print('WARNING: Gauss polynomial has more than 1 real, positive solution')
        print('gauss_poly_coeffs = ', gauss_poly_coeffs)
        print('gauss_poly_roots = ', gauss_poly_roots)
        print('len(rt_indx[0]) = ', len(rt_indx[0]))
        print('np.real(gauss_poly_roots[rt_indx[0]]) = ', np.real(gauss_poly_roots[rt_indx[0]]))
        print('r2_root_ind = ', r2_root_ind)

    # if r2_root_ind==0:
    #     # r2_star = np.real(gauss_poly_roots[rt_indx[0][len(rt_indx[0])-1]])
    #     r2_star = np.real(gauss_poly_roots[rt_indx[0][0]])
    # else:
    #     # r2_star = np.real(gauss_poly_roots[rt_indx[0][len(rt_indx[0])-1]])
    r2_star = np.real(gauss_poly_roots[rt_indx[0][r2_root_ind]])
    print('r2_star = ', r2_star)

    # print('r2_star = ', r2_star/au)
    # print('r2_star = ', r2_star)

    num1 = 6.0*(D[2,0]*(tau1/tau3)+D[1,0]*(tau/tau3))*(r2_star**3)+mu*D[2,0]*(tau**2-tau1**2)*(tau1/tau3)
    den1 = 6.0*(r2_star**3)+mu*(tau**2-tau3**2)

    rho_1_ = ((num1/den1)-D[0,0])/D0

    rho_2_ = A+(mu*B)/(r2_star**3)

    num3 = 6.0*(D[0,2]*(tau3/tau1)-D[1,2]*(tau/tau1))*(r2_star**3)+mu*D[0,2]*(tau**2-tau3**2)*(tau3/tau1)
    den3 = 6.0*(r2_star**3)+mu*(tau**2-tau1**2)

    rho_3_ = ((num3/den3)-D[2,2])/D0

    # print('rho_1_ = ', rho_1_/au,'au')
    # print('rho_2_ = ', rho_2_/au,'au')
    # print('rho_3_ = ', rho_3_/au,'au')
    # print('rho_1_ = ', rho_1_)
    # print('rho_2_ = ', rho_2_)
    # print('rho_3_ = ', rho_3_)

    r1 = R[0]+rho_1_*rho1
    r2 = R[1]+rho_2_*rho2
    r3 = R[2]+rho_3_*rho3

    # print('r1 = ', r1)
    # print('r2 = ', r2)
    # print('r3 = ', r3)

    # print('|r1| = ', np.linalg.norm(r1, ord=2)/au, 'au')
    # print('|r2| = ', np.linalg.norm(r2, ord=2)/au, 'au')
    # print('|r3| = ', np.linalg.norm(r3, ord=2)/au, 'au')

    f1 = lagrangef(mu, r2_star, tau1)
    f3 = lagrangef(mu, r2_star, tau3)

    g1 = lagrangeg(mu, r2_star, tau1)
    g3 = lagrangeg(mu, r2_star, tau3)

    # print('f1 = ', f1)
    # print('f3 = ', f3)
    # print('g1 = ', g1)
    # print('g3 = ', g3)

    v2 = (-f3*r1+f1*r3)/(f1*g3-f3*g1)

    # print('v2 = ', v2)

    return r1, r2, r3, v2, D, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_

# Refinement stage of Gauss method
# INPUT: tau1, tau3, r2, v2, mu, atol, D, R, rho1, rho2, rho3, f_1, g_1, f_3, g_3
# OUTPUT: updated r1, r2, r3, v2, rho_1_, rho_2_, rho_3_, f1, g1, f3, g3
def gauss_refinement(mu, tau1, tau3, r2, v2, atol, D, R, rho1, rho2, rho3, f_1, g_1, f_3, g_3):
    xi1 = univkepler(tau1, r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu, iters=10, atol=atol)
    xi3 = univkepler(tau3, r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu, iters=10, atol=atol)

    r0_ = np.sqrt((r2[0]**2)+(r2[1]**2)+(r2[2]**2))
    v20_ = (v2[0]**2)+(v2[1]**2)+(v2[2]**2)
    alpha0_ = (2.0/r0_)-(v20_/mu)

    z1_ = alpha0_*(xi1**2)
    # print('xi1 = ', xi1)
    # print('xi3 = ', xi3)
    # print('alpha0_ = ', alpha0_)
    # print('z1_ = ', z1_)
    # print('np.cosh(z1_) = ', np.cosh(z1_))
    f1_ = (f_1+lagrangef_(xi1, z1_, r0_))/2
    g1_ = (g_1+lagrangeg_(tau1, xi1, z1_, mu))/2

    z3_ = alpha0_*(xi3**2)
    f3_ = (f_3+lagrangef_(xi3, z3_, r0_))/2
    g3_ = (g_3+lagrangeg_(tau3, xi3, z3_, mu))/2

    # print('f_1 = ', f_1)
    # print('lagrangef_(xi1, z1_, r0_) = ', lagrangef_(xi1, z1_, r0_))
    # print('f1_ = ', f1_)
    # print('g1 = ', g1)
    # print('g1_ = ', g1_)

    # print('f3 = ', f3)
    # print('f3_ = ', f3_)
    # print('g3 = ', g3)
    # print('g3_ = ', g3_)

    denum = f1_*g3_-f3_*g1_

    c1_ = g3_/denum
    c3_ = -g1_/denum

    # print('c1_ = ', c1_)
    # print('c3_ = ', c3_)

    D0  = np.dot(rho1, np.cross(rho2, rho3))

    rho_1_ = (-D[0,0]+D[1,0]/c1_-D[2,0]*(c3_/c1_))/D0
    rho_2_ = (-c1_*D[0,1]+D[1,1]-c3_*D[2,1])/D0
    rho_3_ = (-D[0,2]*(c1_/c3_)+D[1,2]/c3_-D[2,2])/D0

    # print('rho_1_ = ', rho_1_)
    # print('rho_2_ = ', rho_2_)
    # print('rho_3_ = ', rho_3_)

    # print('xi1 xi3 f1 g1 f3 g3 rho1 rho2 rho3')
    # print(xi1, ' ', xi3, ' ', f1_, ' ', g1_, ' ', f3_, ' ', g3_, ' ', rho_1_, ' ', rho_2_, ' ', rho_3_)

    r1 = R[0]+rho_1_*rho1
    r2 = R[1]+rho_2_*rho2
    r3 = R[2]+rho_3_*rho3

    # print('r1 = ', r1)
    # print('r2 = ', r2)
    # print('r3 = ', r3)

    v2 = (-f3_*r1+f1_*r3)/denum

    # print('v2 = ', v2)

    return r1, r2, r3, v2, rho_1_, rho_2_, rho_3_, f1_, g1_, f3_, g3_

# Implementation of Gauss method for MPC optical observations of NEAs
def gauss_estimate_mpc(mpc_object_data, mpc_observatories_data, inds, r2_root_ind=0):
    # mu_Sun = 0.295912208285591100E-03 # Sun's G*m, au^3/day^2
    mu = mu_Sun # cts.GM_sun.to(uts.Unit("au3 / day2")).value

    # extract observations data
    obs_radec, obs_t, site_codes = get_observations_data(mpc_object_data, inds)

    # compute observer position vectors wrt Sun
    R, Ea_hc_pos = get_observer_pos_wrt_sun(mpc_observatories_data, obs_radec, site_codes)

    # perform core Gauss method
    r1, r2, r3, v2, D, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_ = gauss_method_core(obs_radec, obs_t, R, mu, r2_root_ind=r2_root_ind)

    return r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos, rho_1_, rho_2_, rho_3_, obs_t

# Implementation of Gauss method for IOD-formatted optical observations of Earth satellites
def gauss_estimate_sat(iod_object_data, sat_observatories_data, inds, r2_root_ind=0):
    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    mu = mu_Earth

    # extract observations data
    obs_radec, obs_t, site_codes = get_observations_data_sat(iod_object_data, inds)

    obs_t_jd = np.array((obs_radec[0].obstime.jd, obs_radec[1].obstime.jd, obs_radec[2].obstime.jd))

    # print('obs_radec = ', obs_radec)
    # print('obs_t = ', obs_t)
    # print('obs_t_jd = ', obs_t_jd)
    # print('site_codes = ', site_codes)

    # compute observer position vectors wrt Sun
    R = get_observer_pos_wrt_earth(sat_observatories_data, obs_radec, site_codes)

    # perform core Gauss method
    r1, r2, r3, v2, D, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_ = gauss_method_core(obs_radec, obs_t, R, mu, r2_root_ind=r2_root_ind)

    return r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_, obs_t_jd

# Implementation of Gauss method for ra-dec observations of Earth satellites; method for book examples
def gauss_estimate_sat_book(phi_deg, altitude_km, f, ra_hrs, dec_deg, lst_deg, t_sec, r2_root_ind=0):
    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    mu = mu_Earth

    # construct vector of observation time intervals (seconds)
    timeobs = np.zeros((3,), dtype=Time)
    timeobs[0] = Time( datetime(2010, 1, 1) + timedelta(seconds=t_sec[0]) )
    timeobs[1] = Time( datetime(2010, 1, 1) + timedelta(seconds=t_sec[1]) )
    timeobs[2] = Time( datetime(2010, 1, 1) + timedelta(seconds=t_sec[2]) )
    obs_t = np.zeros((3,))
    obs_t[0] = (timeobs[0]-timeobs[1]).to(uts.second).value
    obs_t[1] = 0.0
    obs_t[2] = (timeobs[2]-timeobs[1]).to(uts.second).value

    # construct SkyCoord 3-element array with observational information
    obs_radec = np.zeros((3,), dtype=SkyCoord)
    obs_radec[0] = SkyCoord( ra=ra_hrs[0], dec=dec_deg[0], unit=(uts.hourangle, uts.deg), obstime=timeobs[0])
    obs_radec[1] = SkyCoord( ra=ra_hrs[1], dec=dec_deg[1], unit=(uts.hourangle, uts.deg), obstime=timeobs[1])
    obs_radec[2] = SkyCoord( ra=ra_hrs[2], dec=dec_deg[2], unit=(uts.hourangle, uts.deg), obstime=timeobs[2])

    # compute geocentric observer position vectors at observation event
    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))
    R[0] = observerpos_sat_book(phi_deg, altitude_km, f, lst_deg[0])
    R[1] = observerpos_sat_book(phi_deg, altitude_km, f, lst_deg[1])
    R[2] = observerpos_sat_book(phi_deg, altitude_km, f, lst_deg[2])
    # print('R[0] = ', R[0])
    # print('R[1] = ', R[1])
    # print('R[2] = ', R[2])

    # perform core Gauss method
    r1, r2, r3, v2, D, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_ = gauss_method_core(obs_radec, obs_t, R, mu, r2_root_ind=r2_root_ind)

    return r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_

def gauss_iterator_sat_book(phi_deg, altitude_km, f, ra_hrs, dec_deg, lst_deg, t_sec, refiters=0):
    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    mu = mu_Earth
    r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_ = gauss_estimate_sat_book(phi_deg, altitude_km, f, ra_hrs, dec_deg, lst_deg, t_sec)
    # Apply refinement to Gauss' method, `refiters` iterations
    for i in range(0, refiters):
        r1, r2, r3, v2, rho_1_, rho_2_, rho_3_, f1, g1, f3, g3 = gauss_refinement(mu, tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)
    return r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_

def gauss_iterator_sat(iod_object_data, sat_observatories_data, inds_, refiters=0, r2_root_ind=0):
    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    mu = mu_Earth
    r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_, obs_t = gauss_estimate_sat(iod_object_data, sat_observatories_data, inds_, r2_root_ind=r2_root_ind)
    # Apply refinement to Gauss' method, `refiters` iterations
    for i in range(0,refiters):
        r1, r2, r3, v2, rho_1_, rho_2_, rho_3_, f1, g1, f3, g3 = gauss_refinement(mu, tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)
    return r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_, obs_t

def gauss_iterator_mpc(mpc_object_data, mpc_observatories_data, inds_, refiters=0, r2_root_ind=0):
    # mu_Sun = 0.295912208285591100E-03 # Sun's G*m, au^3/day^2
    mu = mu_Sun # cts.GM_sun.to(uts.Unit("au3 / day2")).value
    r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos, rho_1_, rho_2_, rho_3_, obs_t = gauss_estimate_mpc(mpc_object_data, mpc_observatories_data, inds_, r2_root_ind=r2_root_ind)
    # Apply refinement to Gauss' method, `refiters` iterations
    for i in range(0,refiters):
        r1, r2, r3, v2, rho_1_, rho_2_, rho_3_, f1, g1, f3, g3 = gauss_refinement(mu, tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)
    return r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_, Ea_hc_pos, obs_t

# compute auxiliary vector of observed ra,dec values
# inds = obs_arr
def radec_obs_vec(inds, iod_object_data):
    rov = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1], minutes=1.0*iod_object_data['min'][indm1], seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1], iod_object_data['month'][indm1], iod_object_data['day'][indm1]) + td )
        raHHMMmmm  = iod_object_data['raHH' ][indm1] + (iod_object_data['raMM' ][indm1]+iod_object_data['rammm' ][indm1]/1000.0)/60.0
        decDDMMmmm = iod_object_data['decDD'][indm1] + (iod_object_data['decMM'][indm1]+iod_object_data['decmmm'][indm1]/1000.0)/60.0
        obs_t_ra_dec = SkyCoord(ra=raHHMMmmm, dec=decDDMMmmm, unit=(uts.hourangle, uts.deg), obstime=timeobs)
        rov[2*i-2], rov[2*i-1] = obs_t_ra_dec.ra.rad, obs_t_ra_dec.dec.rad
    return rov

# compute residuals vector for ra/dec observations with pre-computed observed radec values vector
# inds = obs_arr
def radec_res_vec_rov(x, inds, iod_object_data, sat_observatories_data, rov):
    rv = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1], minutes=1.0*iod_object_data['min'][indm1], seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1], iod_object_data['month'][indm1], iod_object_data['day'][indm1]) + td )
        site_code = iod_object_data['station'][indm1]
        obsite = get_station_data(site_code, sat_observatories_data)
        # object position wrt to Earth
        xyz_obj = orbel2xyz(timeobs.jd, cts.GM_earth.to(uts.Unit('km3 / day2')).value, x[0], x[1], x[2], x[3], x[4], x[5])
        # observer position wrt to Earth
        xyz_oe = observerpos_sat(obsite['Latitude'], obsite['Longitude'], obsite['Elev'], timeobs)
        # object position wrt observer (unnormalized LOS vector)
        rho_vec = xyz_obj - xyz_oe
        # compute normalized LOS vector
        rho_vec_norm = np.linalg.norm(rho_vec, ord=2)
        rho_vec_unit = rho_vec/rho_vec_norm
        # compute RA, Dec
        cosd_cosa = rho_vec_unit[0]
        cosd_sina = rho_vec_unit[1]
        sind = rho_vec_unit[2]
        # make sure computed RA (ra_comp) is always within [0.0, 2.0*np.pi]
        ra_comp = np.mod(np.arctan2(cosd_sina, cosd_cosa), 2.0*np.pi)
        dec_comp = np.arcsin(sind)
        #compute angle difference, taking always the smallest difference
        diff_ra = angle_diff_rad(rov[2*i-2], ra_comp)
        diff_dec = angle_diff_rad(rov[2*i-1], dec_comp)
        # compute O-C residual (O-C = "Observed minus Computed")
        rv[2*i-2], rv[2*i-1] = diff_ra, diff_dec
    return rv

# compute residuals vector for ra/dec observations; return observation times and residual vector
# inds = obs_arr
def t_radec_res_vec(x, inds, iod_object_data, sat_observatories_data, rov):
    rv = np.zeros((2*len(inds)))
    tv = np.zeros((len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1], minutes=1.0*iod_object_data['min'][indm1], seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1], iod_object_data['month'][indm1], iod_object_data['day'][indm1]) + td )
        t_jd = timeobs.jd
        site_code = iod_object_data['station'][indm1]
        obsite = get_station_data(site_code, sat_observatories_data)
        # object position wrt to Earth
        xyz_obj = orbel2xyz(t_jd, cts.GM_earth.to(uts.Unit('km3 / day2')).value, x[0], x[1], x[2], x[3], x[4], x[5])
        # observer position wrt to Earth
        xyz_oe = observerpos_sat(obsite['Latitude'], obsite['Longitude'], obsite['Elev'], timeobs)
        # object position wrt observer (unnormalized LOS vector)
        rho_vec = xyz_obj - xyz_oe
        # compute normalized LOS vector
        rho_vec_norm = np.linalg.norm(rho_vec, ord=2)
        rho_vec_unit = rho_vec/rho_vec_norm
        # compute RA, Dec
        cosd_cosa = rho_vec_unit[0]
        cosd_sina = rho_vec_unit[1]
        sind = rho_vec_unit[2]
        # make sure computed RA (ra_comp) is always within [0.0, 2.0*np.pi]
        ra_comp = np.mod(np.arctan2(cosd_sina, cosd_cosa), 2.0*np.pi)
        dec_comp = np.arcsin(sind)
        #compute angle difference, taking always the smallest difference
        diff_ra = angle_diff_rad(rov[2*i-2], ra_comp)
        diff_dec = angle_diff_rad(rov[2*i-1], dec_comp)
        # compute O-C residual (O-C = "Observed minus Computed")
        rv[2*i-2], rv[2*i-1] = diff_ra, diff_dec
        tv[i] = t_jd
    return tv, rv

def gauss_method_mpc(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=0, plot=True):
    # load MPC data for a given NEA
    mpc_object_data = load_mpc_data(body_fname_str)
    # print('MPC observation data:\n', mpc_object_data[ inds ], '\n')

    #load MPC data of listed observatories (longitude, parallax constants C, S)
    mpc_observatories_data = load_mpc_observatories_data('mpc_observatories.txt')

    #definition of the astronomical unit in km
    # au = cts.au.to(uts.Unit('km')).value

    # Sun's G*m value
    # mu_Sun = 0.295912208285591100E-03 # au^3/day^2
    mu = mu_Sun # cts.GM_sun.to(uts.Unit("au3 / day2")).value

    #the total number of observations used
    nobs = len(obs_arr)

    print('nobs = ', nobs)
    print('obs_arr = ', obs_arr)

    #auxiliary arrays
    x_vec = np.zeros((nobs,))
    y_vec = np.zeros((nobs,))
    z_vec = np.zeros((nobs,))
    a_vec = np.zeros((nobs-2,))
    e_vec = np.zeros((nobs-2,))
    taup_vec = np.zeros((nobs-2,))
    I_vec = np.zeros((nobs-2,))
    W_vec = np.zeros((nobs-2,))
    w_vec = np.zeros((nobs-2,))
    n_vec = np.zeros((nobs-2,))
    x_Ea_vec = np.zeros((nobs-2,))
    y_Ea_vec = np.zeros((nobs-2,))
    z_Ea_vec = np.zeros((nobs-2,))
    t_vec = np.zeros((nobs,))

    print('r2_root_ind_vec = ', r2_root_ind_vec)
    print('len(range (0,nobs-2)) = ', len(range (0,nobs-2)))

    for j in range (0,nobs-2):
        # Apply Gauss method to three elements of data
        inds_ = [obs_arr[j]-1, obs_arr[j+1]-1, obs_arr[j+2]-1]
        print('j = ', j)
        r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_, Ea_hc_pos, obs_t = gauss_iterator_mpc(mpc_object_data, mpc_observatories_data, inds_, refiters=refiters, r2_root_ind=r2_root_ind_vec[j])

        # print('|r1| = ', np.linalg.norm(r1,ord=2))
        # print('|r2| = ', np.linalg.norm(r2,ord=2))
        # print('|r3| = ', np.linalg.norm(r3,ord=2))
        # print('r2 = ', r2)
        # print('obs_t[1] = ', obs_t[1])
        # print('v2 = ', v2)

        if j==0:
            t_vec[0] = obs_t[0]
            x_vec[0], y_vec[0], z_vec[0] = np.matmul(rot_equat_to_eclip, r1)
        if j==nobs-3:
            t_vec[nobs-1] = obs_t[2]
            x_vec[nobs-1], y_vec[nobs-1], z_vec[nobs-1] = np.matmul(rot_equat_to_eclip, r3)

        r2_eclip = np.matmul(rot_equat_to_eclip, r2)
        v2_eclip = np.matmul(rot_equat_to_eclip, v2)

        a_num = semimajoraxis(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2], mu)
        e_num = eccentricity(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2], mu)
        f_num = trueanomaly(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2], mu)
        n_num = meanmotion(mu, a_num)

        a_vec[j] = a_num
        e_vec[j] = e_num
        taup_vec[j] = taupericenter(obs_t[1], e_num, f_num, n_num)
        w_vec[j] = np.rad2deg( argperi(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2], mu) )
        I_vec[j] = np.rad2deg( inclination(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2]) )
        W_vec[j] = np.rad2deg( longascnode(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2]) )
        n_vec[j] = n_num
        t_vec[j+1] = obs_t[1]
        x_vec[j+1] = r2_eclip[0]
        y_vec[j+1] = r2_eclip[1]
        z_vec[j+1] = r2_eclip[2]
        Ea_hc_pos_eclip = np.matmul(rot_equat_to_eclip, Ea_hc_pos[1])
        x_Ea_vec[j] = Ea_hc_pos_eclip[0]
        y_Ea_vec[j] = Ea_hc_pos_eclip[1]
        z_Ea_vec[j] = Ea_hc_pos_eclip[2]

    # print(a_num/au, 'au', ', ', e_num)
    # print(a_num, 'au', ', ', e_num)
    # print('j = ', j, 'obs_arr[j] = ', obs_arr[j])

    print('x_vec = ', x_vec)
    # print('a_vec = ', a_vec)
    # print('e_vec = ', e_vec)
    print('a_vec = ', a_vec)
    print('len(a_vec) = ', len(a_vec))
    print('len(a_vec[a_vec>0.0]) = ', len(a_vec[a_vec>0.0]))

    print('e_vec = ', e_vec)
    print('len(e_vec) = ', len(e_vec))
    e_vec_fil1 = e_vec[e_vec<1.0]
    e_vec_fil2 = e_vec_fil1[e_vec_fil1>0.0]
    print('len(e_vec[e_vec<1.0]) = ', len(e_vec_fil2))

    # print('taup_vec = ', taup_vec)
    print('t_vec = ', t_vec)

    a_mean = np.mean(a_vec) #au
    e_mean = np.mean(e_vec) #dimensionless
    taup_mean = np.mean(taup_vec) #deg
    w_mean = np.mean(w_vec) #deg
    I_mean = np.mean(I_vec) #deg
    W_mean = np.mean(W_vec) #deg
    n_mean = np.mean(n_vec) #sec

    print('\nObservational arc:')
    print('Number of observations: ', len(obs_arr))
    print('First observation (UTC) : ', Time(t_vec[0], format='jd').iso)
    print('Last observation (UTC) : ', Time(t_vec[-1], format='jd').iso)

    print('\n*** AVERAGE ORBITAL ELEMENTS (ECLIPTIC, MEAN J2000.0): a, e, taup, omega, I, Omega, T ***')
    print('Semi-major axis (a):                 ', a_mean, 'au')
    print('Eccentricity (e):                    ', e_mean)
    print('Time of pericenter passage (tau):    ', Time(taup_mean, format='jd').iso, 'JDTDB')
    print('Argument of pericenter (omega):      ', w_mean, 'deg')
    print('Inclination (I):                     ', I_mean, 'deg')
    print('Longitude of Ascending Node (Omega): ', W_mean, 'deg')
    print('Orbital period (T):                  ', 2.0*np.pi/n_mean, 'days')

    npoints = 1000
    theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
    t_Ea_vec = np.linspace(2451544.5, 2451544.5+365.3, npoints)
    x_orb_vec = np.zeros((npoints,))
    y_orb_vec = np.zeros((npoints,))
    z_orb_vec = np.zeros((npoints,))
    x_Ea_orb_vec = np.zeros((npoints,))
    y_Ea_orb_vec = np.zeros((npoints,))
    z_Ea_orb_vec = np.zeros((npoints,))

    for i in range(0,npoints):
        x_orb_vec[i], y_orb_vec[i], z_orb_vec[i] = xyz_frame_(a_mean, e_mean, theta_vec[i], np.deg2rad(w_mean), np.deg2rad(I_mean), np.deg2rad(W_mean))
        xyz_Ea_orb_vec_equat = earth_ephemeris(t_Ea_vec[i])/au
        xyz_Ea_orb_vec_eclip = np.matmul(rot_equat_to_eclip, xyz_Ea_orb_vec_equat)
        x_Ea_orb_vec[i], y_Ea_orb_vec[i], z_Ea_orb_vec[i] = xyz_Ea_orb_vec_eclip

    # PLOT
    if plot:
        ax = plt.axes(aspect='equal', projection='3d')

        # Sun-centered orbits: Computed orbit and Earth's
        ax.scatter3D(0.0, 0.0, 0.0, color='yellow', label='Sun')
        ax.scatter3D(x_Ea_vec, y_Ea_vec, z_Ea_vec, color='blue', marker='.', label='Earth orbit')
        ax.plot3D(x_Ea_orb_vec, y_Ea_orb_vec, z_Ea_orb_vec, color='blue', linewidth=0.5)
        ax.scatter3D(x_vec, y_vec, z_vec, color='red', marker='+', label=body_name_str+' orbit')
        ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'red', linewidth=0.5)
        plt.legend()
        ax.set_xlabel('x (au)')
        ax.set_ylabel('y (au)')
        ax.set_zlabel('z (au)')
        xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
        ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) #, ncol=3)
        ax.set_title('Angles-only orbit determ. (Gauss): '+body_name_str)
        plt.show()

    # return x_vec, y_vec, z_vec, x_Ea_vec, y_Ea_vec, z_Ea_vec, a_vec, e_vec, I_vec, W_vec, w_vec
    return a_mean, e_mean, taup_mean, w_mean, I_mean, W_mean, 2.0*np.pi/n_mean

def gauss_method_sat(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=0, plot=True):
    # load IOD data for a given satellite
    iod_object_data = load_iod_data(body_fname_str)
    # print('IOD observation data:\n', iod_object_data, '\n')
    # print('IOD observation data:\n', iod_object_data[ np.array(obs_arr)-1 ], '\n')

    #load data of listed observatories (longitude, latitude, elevation)
    sat_observatories_data = load_sat_observatories_data('sat_tracking_observatories.txt')
    # print('sat_observatories_data = ', sat_observatories_data)

    # Earth's G*m value
    mu = mu_Earth

    # #the total number of observations used
    nobs = len(obs_arr)

    print('nobs = ', nobs)
    print('obs_arr = ', obs_arr)

    # #auxiliary arrays
    x_vec = np.zeros((nobs,))
    y_vec = np.zeros((nobs,))
    z_vec = np.zeros((nobs,))
    a_vec = np.zeros((nobs-2,))
    e_vec = np.zeros((nobs-2,))
    taup_vec = np.zeros((nobs-2,))
    I_vec = np.zeros((nobs-2,))
    W_vec = np.zeros((nobs-2,))
    w_vec = np.zeros((nobs-2,))
    n_vec = np.zeros((nobs-2,))
    t_vec = np.zeros((nobs,))

    print('r2_root_ind_vec = ', r2_root_ind_vec)
    print('len(range (0,nobs-2)) = ', len(range (0,nobs-2)))

    for j in range (0,nobs-2):
        # Apply Gauss method to three elements of data
        inds_ = [obs_arr[j]-1, obs_arr[j+1]-1, obs_arr[j+2]-1]
        # print('inds_ = ', inds_)
        print('j = ', j)
        r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_, obs_t = gauss_iterator_sat(iod_object_data, sat_observatories_data, inds_, refiters=refiters, r2_root_ind=r2_root_ind_vec[j])
        # print('obs_t = ', obs_t)

        # print('|r1| = ', np.linalg.norm(r1,ord=2))
        # print('|r2| = ', np.linalg.norm(r2,ord=2))
        # print('|r3| = ', np.linalg.norm(r3,ord=2))
        # print('r2 = ', r2)
        # print('obs_t[1] = ', obs_t[1])
        # print('v2 = ', v2)

        if j==0:
            t_vec[0] = obs_t[0]
            x_vec[0] = r1[0]
            y_vec[0] = r1[1]
            z_vec[0] = r1[2]
        if j==nobs-3:
            t_vec[nobs-1] = obs_t[2]
            x_vec[nobs-1] = r3[0]
            y_vec[nobs-1] = r3[1]
            z_vec[nobs-1] = r3[2]

        a_num = semimajoraxis(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
        e_num = eccentricity(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
        f_num = trueanomaly(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
        n_num = meanmotion(mu, a_num)

        a_vec[j] = a_num
        e_vec[j] = e_num
        # print('obst_t = ', obs_t)
        # print('obs_t[1] = ', obs_t[1])
        taup_vec[j] = taupericenter(obs_t[1], e_num, f_num, n_num*86400)
        w_vec[j] = np.rad2deg( argperi(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu) )
        I_vec[j] = np.rad2deg( inclination(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
        W_vec[j] = np.rad2deg( longascnode(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
        n_vec[j] = n_num
        t_vec[j+1] = obs_t[1]
        x_vec[j+1] = r2[0]
        y_vec[j+1] = r2[1]
        z_vec[j+1] = r2[2]

        # print(a_num, 'km', ', ', e_num)
        # print('n_num = ', n_num, ' T_num = ', 2.0*np.pi/n_num)
        # # print('j = ', j, 'obs_arr[j] = ', obs_arr[j])

    # # print('x_vec = ', x_vec)
    print('a_vec = ', a_vec)
    print('len(a_vec) = ', len(a_vec))
    # print('len(a_vec[a_vec>0.0]) = ', len(a_vec[a_vec>0.0]))

    print('e_vec = ', e_vec)
    print('len(e_vec) = ', len(e_vec))
    # e_vec_fil1 = e_vec[e_vec<1.0]
    # e_vec_fil2 = e_vec_fil1[e_vec_fil1>0.0]
    # print('len(e_vec[e_vec<1.0]) = ', len(e_vec_fil2))

    # print('w_vec = ', w_vec)
    # print('I_vec = ', I_vec)
    # print('W_vec = ', W_vec)
    # print('taup_vec = ', taup_vec)
    # print('t_vec = ', t_vec)

    a_mean = np.mean(a_vec) #km
    e_mean = np.mean(e_vec) #dimensionless
    taup_mean = np.mean(taup_vec) #deg
    w_mean = np.mean(w_vec) #deg
    I_mean = np.mean(I_vec) #deg
    W_mean = np.mean(W_vec) #deg
    n_mean = np.mean(n_vec) #sec

    print('\nObservational arc:')
    print('Number of observations: ', len(obs_arr))
    print('First observation (UTC) : ', Time(t_vec[0], format='jd').iso)
    print('Last observation (UTC) : ', Time(t_vec[-1], format='jd').iso)

    print('\n*** AVERAGE ORBITAL ELEMENTS (EQUATORIAL): a, e, taup, omega, I, Omega, T ***')
    print('Semi-major axis (a):                 ', a_mean, 'km')
    print('Eccentricity (e):                    ', e_mean)
    print('Time of pericenter passage (tau):    ', Time(taup_mean, format='jd').iso, 'JDUTC')
    print('Argument of pericenter (omega):      ', w_mean, 'deg')
    print('Inclination (I):                     ', I_mean, 'deg')
    print('Longitude of Ascending Node (Omega): ', W_mean, 'deg')
    print('Orbital period (T):                  ', 2.0*np.pi/n_mean/60.0, 'min')

    npoints = 1000
    theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
    x_orb_vec = np.zeros((npoints,))
    y_orb_vec = np.zeros((npoints,))
    z_orb_vec = np.zeros((npoints,))

    for i in range(0,npoints):
        x_orb_vec[i], y_orb_vec[i], z_orb_vec[i] = xyz_frame_(a_mean, e_mean, theta_vec[i], np.deg2rad(w_mean), np.deg2rad(I_mean), np.deg2rad(W_mean))

    # print('x_vec = ', x_vec)
    # print('y_vec = ', y_vec)
    # print('z_vec = ', z_vec)

    # PLOT
    if plot:
        ax = plt.axes(aspect='equal', projection='3d')

        # Earth-centered orbits: Computed orbit and Earth's
        ax.scatter3D(0.0, 0.0, 0.0, color='blue', label='Earth')
        ax.scatter3D(x_vec, y_vec, z_vec, color='red', marker='+', label=body_name_str+' orbit')
        ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'red', linewidth=0.5)
        plt.legend()
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
        ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) #, ncol=3)
        ax.set_title('Angles-only orbit determ. (Gauss): '+body_name_str)
        plt.show()

    # # return x_vec, y_vec, z_vec, x_Ea_vec, y_Ea_vec, z_Ea_vec, a_vec, e_vec, I_vec, W_vec, w_vec
    return a_mean, e_mean, taup_mean, w_mean, I_mean, W_mean, 2.0*np.pi/n_mean/60.0

def gauss_LS_sat(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, gaussiters=0, plot=True):

    # load IOD data for a given satellite
    iod_object_data = load_iod_data(body_fname_str)
    # print('len(iod_object_data) = ', len(iod_object_data))
    # print('IOD observation data:\n', iod_object_data[ np.array(obs_arr)-1 ], '\n')

    #load data of listed observatories (longitude, latitude, elevation)
    sat_observatories_data = load_sat_observatories_data('sat_tracking_observatories.txt')
    # print('sat_observatories_data = ', sat_observatories_data)

    #get preliminary orbit using Gauss method
    #q0 : a, e, taup, I, W, w, T
    q0 = np.array(gauss_method_sat(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=gaussiters, plot=False))
    x0 = q0[0:6]
    x0[3:6] = np.deg2rad(x0[3:6])

    obs_arr_ls = np.array(range(1, len(iod_object_data)+1))
    print('obs_arr_ls = ', obs_arr_ls)
    # print('obs_arr_ls[0] = ', obs_arr_ls[0])
    # print('obs_arr_ls[-1] = ', obs_arr_ls[-1])
    # nobs_ls = len(obs_arr_ls)
    # print('nobs_ls = ', nobs_ls)

    rov = radec_obs_vec(obs_arr_ls, iod_object_data)
    print('rov = ', rov)
    print('len(rov) = ', len(rov))

    rv0 = radec_res_vec_rov(x0, obs_arr_ls, iod_object_data, sat_observatories_data, rov)
    Q0 = np.linalg.norm(rv0, ord=2)/len(rv0)

    print('rv0 = ', rv0)
    print('Q0 = ', Q0)

    Q_ls = least_squares(radec_res_vec_rov, x0, args=(obs_arr_ls, iod_object_data, sat_observatories_data, rov), method='lm', xtol=1e-13)

    print('INFO: scipy.optimize.least_squares exited with code', Q_ls.status)
    print(Q_ls.message,'\n')
    print('Q_ls.x = ', Q_ls.x)

    tv_star, rv_star = t_radec_res_vec(Q_ls.x, obs_arr_ls, iod_object_data, sat_observatories_data, rov)
    Q_star = np.linalg.norm(rv_star, ord=2)/len(rv_star)
    print('rv* = ', rv_star)
    print('Q* = ', Q_star)

    print('Total residual evaluated at Gauss solution: ', Q0)
    print('Total residual evaluated at least-squares solution: ', Q_star, '\n')
    # # print('Percentage improvement: ', (Q0-Q_star)/Q0*100, ' %')

    print('Observational arc:')
    print('Number of observations: ', len(obs_arr_ls))
    print('First observation (UTC) : ', Time(tv_star[0], format='jd').iso)
    print('Last observation (UTC) : ', Time(tv_star[-1], format='jd').iso)

    n_num = meanmotion(mu_Earth, Q_ls.x[0])

    print('\nOrbital elements, Gauss + least-squares solution:')
    # print('Reference epoch (t0):                ', t_mean)
    print('Semi-major axis (a):                 ', Q_ls.x[0], 'km')
    print('Eccentricity (e):                    ', Q_ls.x[1])
    print('Time of pericenter passage (tau):    ', Time(Q_ls.x[2], format='jd').iso, 'JDUTC')
    # print('Pericenter altitude (q):             ', Q_ls.x[0]*(1.0-Q_ls.x[1])-Re, 'km')
    # print('Apocenter altitude (Q):              ', Q_ls.x[0]*(1.0+Q_ls.x[1])-Re, 'km')
    # print('True anomaly at epoch (f0):          ', np.rad2deg(time2truean(Q_ls.x[0], Q_ls.x[1], mu_Sun, t_mean, Q_ls.x[2])), 'deg')
    print('Argument of pericenter (omega):      ', np.rad2deg(Q_ls.x[3]), 'deg')
    print('Inclination (I):                     ', np.rad2deg(Q_ls.x[4]), 'deg')
    print('Longitude of Ascending Node (Omega): ', np.rad2deg(Q_ls.x[5]), 'deg')
    print('Orbital period (T):                  ', 2.0*np.pi/n_num/60.0, 'min')

    ra_res_vec = np.rad2deg(rv_star[0::2])*(3600.0)
    dec_res_vec = np.rad2deg(rv_star[1::2])*(3600.0)

    # print('len(ra_res_vec) = ', len(ra_res_vec))
    # print('len(dec_res_vec) = ', len(dec_res_vec))
    # print('nobs_ls = ', nobs_ls)
    # print('len(tv_star) = ', len(tv_star))
    # # print('tv_star = ', tv_star)

    # y_rad = 0.001

    # PLOT
    if plot:
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].set_title('Gauss + LS fit residuals: RA, Dec')
        axarr[0].scatter(tv_star, ra_res_vec, s=0.75, label='delta RA (\")')
        axarr[0].set_ylabel('RA (\")')
        axarr[1].scatter(tv_star, dec_res_vec, s=0.75, label='delta Dec (\")')
        axarr[1].set_xlabel('time (JDUTC)')
        axarr[1].set_ylabel('Dec (\")')
        # # plt.xlim(4,5)
        # # plt.ylim(-y_rad, y_rad)
        plt.show()

        npoints = 1000
        theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
        x_orb_vec = np.zeros((npoints,))
        y_orb_vec = np.zeros((npoints,))
        z_orb_vec = np.zeros((npoints,))

        for i in range(0,npoints):
            x_orb_vec[i], y_orb_vec[i], z_orb_vec[i] = xyz_frame_(Q_ls.x[0], Q_ls.x[1], theta_vec[i], Q_ls.x[3], Q_ls.x[4], Q_ls.x[5])

        ax = plt.axes(aspect='equal', projection='3d')

        # Earth-centered orbits: Computed orbit and Earth's
        ax.scatter3D(0.0, 0.0, 0.0, color='blue', label='Earth')
        # ax.scatter3D(x_vec, y_vec, z_vec, color='red', marker='+')
        ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'red', linewidth=0.5, label=body_name_str+' orbit')
        plt.legend()
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
        ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) #, ncol=3)
        ax.set_title('Satellite orbit (Gauss+LS): '+body_name_str)
        plt.show()

    return Q_ls.x[0], Q_ls.x[1], Time(Q_ls.x[2], format='jd'), np.rad2deg(Q_ls.x[3]), np.rad2deg(Q_ls.x[4]), np.rad2deg(Q_ls.x[5]), 2.0*np.pi/n_num/60.0
