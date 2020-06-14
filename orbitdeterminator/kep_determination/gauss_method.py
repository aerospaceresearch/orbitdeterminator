"""Implements Gauss' method for three topocentric right ascension and
declination measurements of celestial bodies. Supports both Earth-centered and
Sun-centered orbits."""

import numpy as np
from astropy.coordinates import Longitude, SkyCoord
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
from astropy import units as uts
from astropy import constants as cts
from astropy.time import Time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from poliastro.stumpff import c2, c3
from astropy.coordinates.earth_orientation import obliquity
from astropy.coordinates.matrix_utilities import rotation_matrix
import argparse

# declare astronomical constants in appropriate units
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

#compute rotation matrices from equatorial to ecliptic frame and viceversa
obliquity_j2000 = obliquity(2451544.5) # mean obliquity of the ecliptic at J2000.0
rot_equat_to_eclip = rotation_matrix( obliquity_j2000, 'x') #rotation matrix from equatorial to ecliptic frames
rot_eclip_to_equat = rotation_matrix(-obliquity_j2000, 'x') #rotation matrix from ecliptic to equatorial frames

# set output from printed arrays at full precision
np.set_printoptions(precision=16)

# convention:
# a: semi-major axis
# e: eccentricity
# taup: time of pericenter passage
# Euler angles:
# omega: argument of pericenter
# I: inclination
# Omega: longitude of ascending node

#rotation about the z-axis about an angle `ang`
def rotz(ang):
    cos_ang = np.cos(ang)
    sin_ang = np.sin(ang)
    return np.array(((cos_ang,-sin_ang,0.0), (sin_ang, cos_ang,0.0), (0.0,0.0,1.0)))

#rotation about the x-axis about an angle `ang`
def rotx(ang):
    cos_ang = np.cos(ang)
    sin_ang = np.sin(ang)
    return np.array(((1.0,0.0,0.0), (0.0,cos_ang,-sin_ang), (0.0,sin_ang,cos_ang)))

#rotation from the orbital plane to the inertial frame
#it is composed of the following rotations, in that order:
#1) rotation about the z axis about an angle `omega` (argument of pericenter)
#2) rotation about the x axis about an angle `I` (inclination)
#3) rotation about the z axis about an angle `Omega` (longitude of ascending node)
def orbplane2frame_(omega,I,Omega):
    P2_mul_P3 = np.matmul(rotx(I),rotz(omega))
    return np.matmul(rotz(Omega),P2_mul_P3)

def orbplane2frame(x):
    return orbplane2frame_(x[0],x[1],x[2])

# get Keplerian range
def kep_r_(a, e, f):
    return a*(1.0-e**2)/(1.0+e*np.cos(f))

def kep_r(x):
    return kep_r_(x[0],x[1],x[2])

# get cartesian positions wrt orbital plane
def xyz_orbplane_(a, e, f):
    r = kep_r_(a, e, f)
    return np.array((r*np.cos(f),r*np.sin(f),0.0))

def xyz_orbplane(x):
    return xyz_orbplane_(x[0],x[1],x[2])

# get cartesian positions wrt inertial frame from orbital elements
def xyz_frame2(a,e,f,omega,I,Omega):
    return np.matmul( orbplane2frame_(omega,I,Omega) , xyz_orbplane_(a, e, f) )

def xyz_frame(x):
    return np.matmul( orbplane2frame(x[3:6]) , xyz_orbplane(x[0:3]) )

# get mean motion from mass parameter (mu) and semimajor axis (a)
def meanmotion(mu,a):
    return np.sqrt(mu/(a**3))

# get mean anomaly from mean motion (n), time (t) and time of pericenter passage (taup)
def meananomaly(n, t, taup):
    return np.mod(n*(t-taup), 2.0*np.pi)

# compute eccentric anomaly (E) from eccentricity (e) and mean anomaly (M)
def eccentricanomaly(e,M):
    M0 = np.mod(M,2*np.pi)
    E0 = M0 + np.sign(np.sin(M0))*0.85*e #Murray-Dermotts' initial estimate
    # successive approximations via Newtons' method
    for i in range(0,4):
        #TODO: implement modified Newton's method for Kepler's equation (Murray-Dermott)
        Eans = E0 - (E0-e*np.sin(E0)-M0)/(1.0-e*np.cos(E0))
        E0 = Eans
    return E0

# compute true anomaly (f) from eccentricity (e) and eccentric anomaly (E)
def trueanomaly(e,E):
    Enew = np.mod(E,2.0*np.pi)
    return 2.0*np.arctan(  np.sqrt((1.0+e)/(1.0-e))*np.tan(Enew/2)  )

# compute eccentric anomaly (E) from eccentricity (e) and true anomaly (f)
def truean2eccan(e, f):
    fnew = np.mod(f,2.0*np.pi)
    return 2.0*np.arctan(  np.sqrt((1.0-e)/(1.0+e))*np.tan(fnew/2)  )

# compute true anomaly from eccentricity and mean anomaly
def meanan2truean(e,M):
    return trueanomaly(e, eccentricanomaly(e, M))

# compute true anomaly from time, a, e, mu and taup
def time2truean(a, e, mu, t, taup):
    return meanan2truean(e, meananomaly(meanmotion(mu, a), t, taup))

# compute cartesian positions (x,y,z) at time t
# for mass parameter mu from orbital elements a, e, taup, I, omega, Omega
def orbel2xyz(t, mu, a, e, taup, omega, I, Omega):

    # compute true anomaly at time t
    f = time2truean(a, e, mu, t, taup)
    # get cartesian positions wrt inertial frame from orbital elements
    return xyz_frame2(a, e, f, omega, I, Omega)

def load_mpc_observatories_data(mpc_observatories_fname):
    """Load Minor Planet Center observatories data using numpy's genfromtxt function.

       Args:
           mpc_observatories_fname (str): file name with MPC observatories data.

       Returns:
           ndarray: data read from the text file (output from numpy.genfromtxt)
    """
    obs_dt = 'S3, f8, f8, f8, S48'
    obs_delims = [4, 10, 9, 10, 48]
    return np.genfromtxt(mpc_observatories_fname, dtype=obs_dt, names=True, delimiter=obs_delims, autostrip=True, encoding=None)

def load_sat_observatories_data(sat_observatories_fname):
    """Load COSPAR satellite tracking observatories data using numpy's genfromtxt function.

       Args:
           sat_observatories_fname (str): file name with COSPAR observatories data.

       Returns:
           ndarray: data read from the text file (output from numpy.genfromtxt)
    """
    obs_dt = 'i8, S2, f8, f8, f8, S18'
    obs_delims = [4, 3, 10, 10, 8, 21]
    return np.genfromtxt(sat_observatories_fname, dtype=obs_dt, names=True, delimiter=obs_delims, autostrip=True, encoding=None, skip_header=1)

def get_observatory_data(observatory_code, mpc_observatories_data):
    """Load individual data of MPC observatory corresponding to given observatory code.

       Args:
           observatory_code (int): MPC observatory code.
           mpc_observatories_data (string): path to file containing MPC observatories data.

       Returns:
           ndarray: observatory data corresponding to code.
    """
    arr_index = np.where(mpc_observatories_data['Code'] == observatory_code)
    return mpc_observatories_data[arr_index[0][0]]

def get_station_data(station_code, sat_observatories_data):
    """Load individual data of COSPAR satellite tracking observatory corresponding to given observatory code.

       Args:
           observatory_code (int): COSPAR station code.

       Returns:
           ndarray: station data (Lat, Long, Elev) corresponding to observatory code.
    """
    arr_index = np.where(sat_observatories_data['No'] == station_code)
    return sat_observatories_data[arr_index[0][0]]

def load_mpc_data(fname):
    """Loads minor planet position observation data from MPC-formatted files.
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
        x (ndarray): array of minor planet position observations following the
        MPC format.
    """
    # dt is the dtype for MPC-formatted text files
    dt = 'i8,S7,S1,S1,S1,i8,i8,i8,f8,U24,S9,S6,S6,S3'
    # mpc_names correspond to the dtype names of each field
    mpc_names = ['mpnum','provdesig','discovery','publishnote','j2000','yr','month','day','utc','radec','9xblank','magband','6xblank','observatory']
    # mpc_delims are the fixed-width column delimiter following MPC format description
    mpc_delims = [5,7,1,1,1,4,3,3,7,24,9,6,6,3]
    return np.genfromtxt(fname, dtype=dt, names=mpc_names, delimiter=mpc_delims, autostrip=True)

def load_iod_data(fname):
    """ Loads satellite position observation data files following the Interactive
    Orbit Determination format (IOD). Currently, the only supported angle format
    are 1,2,3&7, as specified in IOD format.
    IOD format is described at http://www.satobs.org/position/IODformat.html.

    TODO: convert IOD angle formats 4,5&6 from AZ/EL to RA/DEC.

    Args:
        fname (string): name of the IOD-formatted text file to be parsed

    Returns:
        x (numpy array): array of satellite position observations following the
        IOD format, with angle format code = 2.
    """

    # dt is the dtype for IOD-formatted text files
    dt = 'S15, i8, S1, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, S8, S7, i8, i8, S1, S1, i8, i8, i8'

    # iod_names correspond to the dtype names of each field
    iod_names = ['object', 'station', 'stationstatus',
                  'yr', 'month', 'day',
                  'hr', 'min', 'sec', 'msec', 'timeM', 'timeX',
                  'angformat', 'epoch',
                  'raaz', 'decel', 'radecazelM', 'radecazelX',
                  'optical', 'vismagsign', 'vismag', 'vismaguncertainty', 'flashperiod']

    # iod_delims corresponds to the delimiter for cutting the right variable from each input string
    iod_delims = [15, 5, 2,
                   5, 2, 2,
                   2, 2, 2, 3, 2, 1,
                   2, 1,
                   8, 7, 2, 1,
                   2, 1, 3, 3, 9]

    iod_input_lines = np.genfromtxt(fname, dtype=dt, names=iod_names, delimiter=iod_delims, autostrip=True)

    right_ascension = []
    declination = []
    azimuth = []
    elevation = []

    for i in range(len(iod_input_lines)):

        RA = -1.0
        DEC = -1.0
        AZ = -1.0
        EL = -1.0

        if iod_input_lines["angformat"][i] == 1:
            # 1: RA/DEC = HHMMSSs+DDMMSS MX   (MX in seconds of arc)
            RAAZ = iod_input_lines["raaz"][i].decode()
            HH = float(RAAZ[0:2])
            MM = float(RAAZ[2:4])
            SS = float(RAAZ[4:6])
            s = float(RAAZ[6])
            RA = (HH + (MM + (SS + s / 10.0) / 60.0) / 60.0) / 24.0 * 360.0

            DECEL = iod_input_lines["decel"][i].decode()
            DD = float(DECEL[1:3])
            MM = float(DECEL[3:5])
            SS = float(DECEL[5:7])
            DEC = DD + (MM + SS / 60.0) / 60.0
            if DECEL[0] == "-":
                DEC = -1.0 * DEC

        elif iod_input_lines["angformat"][i] == 2:
            # 2: RA/DEC = HHMMmmm+DDMMmm MX   (MX in minutes of arc)
            RAAZ = iod_input_lines["raaz"][i].decode()
            HH = float(RAAZ[0:2])
            MM = float(RAAZ[2:4])
            mmm = float(RAAZ[4:7])
            RA = (HH + (MM + mmm / 1000.0) / 60.0) / 24.0 * 360.0

            DECEL = iod_input_lines["decel"][i].decode()
            DD = float(DECEL[1:3])
            MM = float(DECEL[3:5])
            mm = float(DECEL[5:7])
            DEC = DD + (MM + mm / 100.0) / 60.0
            if DECEL[0] == "-":
                DEC = -1.0 * DEC

        elif iod_input_lines["angformat"][i] == 3:
            # 3: RA/DEC = HHMMmmm+DDdddd MX   (MX in degrees of arc)
            RAAZ = iod_input_lines["raaz"][i].decode()
            HH = float(RAAZ[0:2])
            MM = float(RAAZ[2:4])
            mmm = float(RAAZ[4:7])
            RA = (HH + (MM + mmm / 1000.0) / 60.0) / 24.0 * 360.0

            DECEL = iod_input_lines["decel"][i].decode()
            DD = float(DECEL[1:3])
            dddd = float(DECEL[3:7])
            DEC = (DD + (dddd / 1000.0))
            if DECEL[0] == "-":
                DEC = -1.0 * DEC

        elif iod_input_lines["angformat"][i] == 4:
            # 4: AZ/EL  = DDDMMSS+DDMMSS MX   (MX in seconds of arc)
            RAAZ = iod_input_lines["raaz"][i].decode()
            DDD = float(RAAZ[0:3])
            MM = float(RAAZ[3:5])
            SS = float(RAAZ[5:7])
            AZ = DDD + (MM + SS / 60.0) / 60.0

            DECEL = iod_input_lines["decel"][i].decode()
            DD = float(DECEL[1:3])
            MM = float(DECEL[3:5])
            SS = float(DECEL[5:7])
            EL = DD + (MM + SS / 60.0) / 60.0
            if DECEL[0] == "-":
                EL = -1.0 * EL

            # TODO: convert from AZ/EL to RA/DEC

        elif iod_input_lines["angformat"][i] == 5:
            # 5: AZ/EL  = DDDMMmm+DDMMmm MX   (MX in minutes of arc)
            RAAZ = iod_input_lines["raaz"][i].decode()
            DDD = float(RAAZ[0:3])
            MM = float(RAAZ[3:5])
            SS = float(RAAZ[5:7])
            AZ = DDD + (MM + SS / 60.0) / 60.0

            DECEL = iod_input_lines["decel"][i].decode()
            DD = float(DECEL[1:3])
            MM = float(DECEL[3:5])
            mm = float(DECEL[5:7])
            EL = DD + (MM + mm / 100.0) / 60.0
            if DECEL[0] == "-":
                EL = -1.0 * EL

            # TODO: convert from AZ/EL to RA/DEC

        elif iod_input_lines["angformat"][i] == 6:
            # 6: AZ/EL  = DDDdddd+DDdddd MX   (MX in degrees of arc)
            RAAZ = iod_input_lines["raaz"][i].decode()
            DDD = float(RAAZ[0:3])
            dddd = float(RAAZ[3:7])
            AZ = DDD + dddd / 1000.0

            DECEL = iod_input_lines["decel"][i].decode()
            DD = float(DECEL[1:3])
            dddd = float(DECEL[3:7])
            EL = DD + dddd / 1000.0
            if DECEL[0] == "-":
                EL = -1.0 * EL

            # TODO: convert from AZ/EL to RA/DEC

        elif iod_input_lines["angformat"][i] == 7:
            # 7: RA/DEC = HHMMSSs+DDdddd MX   (MX in degrees of arc)
            RAAZ = iod_input_lines["raaz"][i].decode()
            HH = float(RAAZ[0:2])
            MM = float(RAAZ[2:4])
            SS = float(RAAZ[4:6])
            s = float(RAAZ[6])
            RA = (HH + (MM + (SS + s / 10.0) / 60.0) / 60.0) / 24.0 * 360.0

            DECEL = iod_input_lines["decel"][i].decode()
            DD = float(DECEL[1:3])
            dddd = float(DECEL[3:7])
            DEC = (DD + (dddd / 1000.0))
            if DECEL[0] == "-":
                DEC = -1.0 * DEC

        #else:
        #    # TODO: when not defined, we assume it is RA/DEC

        right_ascension.append(RA)
        declination.append(DEC)
        azimuth.append(AZ)
        elevation.append(EL)

    # expanding the input iod data with the position data in different formats
    iod = {}
    for name in iod_names:
         iod[name] = iod_input_lines[name].tolist()

    iod["right_ascension"] = right_ascension
    iod["declination"] = declination
    iod["azimuth"] = azimuth
    iod["elevation"] = elevation

    return iod

def observerpos_mpc(long, parallax_s, parallax_c, t_utc):
    """Compute geocentric observer position at UTC instant t_utc, for Sun-centered orbits,
    at a given observation site defined by its longitude, and parallax constants S and C.
    Formula taken from top of page 266, chapter 5, Orbital Mechanics book (Curtis).
    The parallax constants S and C are defined by:
    S=rho cos phi' C=rho sin phi', where
    rho: slant range
    phi': geocentric latitude

       Args:
           long (float): longitude of observing site
           parallax_s (float): parallax constant S of observing site
           parallax_c (float): parallax constant C of observing site
           t_utc (astropy.time.Time): UTC time of observation

       Returns:
           1x3 numpy array: cartesian components of observer's geocentric position
    """
    # Earth's equatorial radius in kilometers
    # Re = cts.R_earth.to(uts.Unit('km')).value

    # define Longitude object for the observation site longitude
    long_site = Longitude(long, uts.degree, wrap_angle=360.0*uts.degree)
    # compute sidereal time of observation at site
    t_site_lmst = t_utc.sidereal_time('mean', longitude=long_site)
    lmst_rad = t_site_lmst.rad # np.deg2rad(lmst_hrs*15.0) # radians

    # compute cartesian components of geocentric (non rotating) observer position
    x_gc = Re*parallax_c*np.cos(lmst_rad)
    y_gc = Re*parallax_c*np.sin(lmst_rad)
    z_gc = Re*parallax_s

    return np.array((x_gc,y_gc,z_gc))

def observerpos_sat(lat, long, elev, t_utc):
    """Compute geocentric observer position at UTC instant t_utc, for Earth-centered orbits,
    at a given observation site defined by its longitude, geodetic latitude and elevation above reference ellipsoid.
    Formula taken from bottom of page 265 (Eq. 5.56), chapter 5, Orbital Mechanics book (Curtis).

       Args:
           lat (float): geodetic latitude (deg)
           long (float): longitude (deg)
           elev (float): elevation above reference ellipsoid (m)
           t_utc (astropy.time.Time): UTC time of observation

       Returns:
           1x3 numpy array: cartesian components of observer's geocentric position
    """

    # Earth's equatorial radius in kilometers
    # Re = cts.R_earth.to(uts.Unit('km')).value

    # define Longitude object for the observation site longitude
    long_site = Longitude(long, uts.degree, wrap_angle=180.0*uts.degree)
    # compute sidereal time of observation at site
    t_site_lmst = t_utc.sidereal_time('mean', longitude=long_site)
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

    return np.array((x_gc,y_gc,z_gc))

def losvector(ra_rad, dec_rad):
    """Compute line-of-sight (LOS) vector for given values of right ascension
    and declination. Both angles must be provided in radians.

       Args:
           ra_rad (float): right ascension (rad)
           dec_rad (float): declination (rad)

       Returns:
           1x3 numpy array: cartesian components of LOS vector.
    """
    cosa_cosd = np.cos(ra_rad)*np.cos(dec_rad)
    sina_cosd = np.sin(ra_rad)*np.cos(dec_rad)
    sind = np.sin(dec_rad)
    return np.array((cosa_cosd, sina_cosd, sind))

def lagrangef(mu, r2, tau):
    """Compute 1st order approximation to Lagrange's f function.

       Args:
           mu (float): gravitational parameter attracting body
           r2 (float): radial distance
           tau (float): time interval

       Returns:
           float: Lagrange's f function value
    """
    return 1.0-0.5*(mu/(r2**3))*(tau**2)

def lagrangeg(mu, r2, tau):
    """Compute 1st order approximation to Lagrange's g function.

       Args:
           mu (float): gravitational parameter attracting body
           r2 (float): radial distance
           tau (float): time interval

       Returns:
           float: Lagrange's g function value
    """
    return tau-(1.0/6.0)*(mu/(r2**3))*(tau**3)

# Set of functions for cartesian states -> Keplerian elements

def kep_h_norm(x, y, z, u, v, w):
    """Compute norm of specific angular momentum vector h.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity

       Returns:
           float: norm of specific angular momentum vector, h.
    """
    return np.sqrt( (y*w-z*v)**2 + (z*u-x*w)**2 + (x*v-y*u)**2 )

def kep_h_vec(x, y, z, u, v, w):
    """Compute specific angular momentum vector h.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity

       Returns:
           float: specific angular momentum vector, h.
    """
    return np.array((y*w-z*v, z*u-x*w, x*v-y*u))

def semimajoraxis(x, y, z, u, v, w, mu):
    """Compute value of semimajor axis, a.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity
           mu (float): gravitational parameter

       Returns:
           float: semimajor axis, a
    """
    myRadius=np.sqrt((x**2)+(y**2)+(z**2))
    myVelSqr=(u**2)+(v**2)+(w**2)
    return 1.0/( (2.0/myRadius)-(myVelSqr/mu) )

def eccentricity(x, y, z, u, v, w, mu):
    """Compute value of eccentricity, e.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity
           mu (float): gravitational parameter

       Returns:
           float: eccentricity, e
    """
    h2 = ((y*w-z*v)**2) + ((z*u-x*w)**2) + ((x*v-y*u)**2)
    a = semimajoraxis(x,y,z,u,v,w,mu)
    quotient = h2/( mu*a )
    return np.sqrt(1.0 - quotient)

def inclination(x, y, z, u, v, w):
    """Compute value of inclination, I.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity

       Returns:
           float: inclination, I
    """
    my_hz = x*v-y*u
    my_h = np.sqrt( (y*w-z*v)**2 + (z*u-x*w)**2 + (x*v-y*u)**2 )

    return np.arccos(my_hz/my_h)

def longascnode(x, y, z, u, v, w):
    """Compute value of longitude of ascending node, computed as
    the angle between x-axis and the vector n = (-hy,hx,0), where hx, hy, are
    respectively, the x and y components of specific angular momentum vector, h.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity

       Returns:
           float: longitude of ascending node
    """
    res = np.arctan2(y*w-z*v, x*w-z*u) # remember atan2 is atan2(y/x)
    if res >= 0.0:
        return res
    else:
        return res+2.0*np.pi

def rungelenz(x, y, z, u, v, w, mu):
    """Compute the cartesian components of Laplace-Runge-Lenz vector.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity
           mu (float): gravitational parameter

       Returns:
           float: Laplace-Runge-Lenz vector
    """
    r = np.sqrt(x**2+y**2+z**2)
    lrl_x = ( -(z*u-x*w)*w+(x*v-y*u)*v )/mu-x/r
    lrl_y = ( -(x*v-y*u)*u+(y*w-z*v)*w )/mu-y/r
    lrl_z = ( -(y*w-z*v)*v+(z*u-x*w)*u )/mu-z/r
    return np.array((lrl_x, lrl_y, lrl_z))

def argperi(x, y, z, u, v, w, mu):
    """Compute the argument of pericenter.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity
           mu (float): gravitational parameter

       Returns:
           float: argument of pericenter
    """
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

def trueanomaly5(x, y, z, u, v, w, mu):
    """Compute the true anomaly from cartesian state.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity
           mu (float): gravitational parameter

       Returns:
           float: true anomaly
    """
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
    """Compute the time of pericenter passage.

       Args:
           t (float): current time
           e (float): eccentricity
           f (float): true anomaly
           n (float): Keplerian mean motion

       Returns:
           float: time of pericenter passage
    """
    E0 = truean2eccan(e, f)
    M0 = np.mod(E0-e*np.sin(E0), 2.0*np.pi)
    return t-M0/n

def alpha(x, y, z, u, v, w, mu):
    """Compute the inverse of the semimajor axis.

       Args:
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity
           mu (float): gravitational parameter

       Returns:
           float: alpha = 1/a
    """
    myRadius=np.sqrt((x**2)+(y**2)+(z**2))
    myVelSqr=(u**2)+(v**2)+(w**2)
    return (2.0/myRadius)-(myVelSqr/mu)

def univkepler(dt, x, y, z, u, v, w, mu, iters=5, atol=1e-15):
    """Compute the current value of the universal Kepler anomaly, xi.

       Args:
           dt (float): time interval
           x (float): x-component of position
           y (float): y-component of position
           z (float): z-component of position
           u (float): x-component of velocity
           v (float): y-component of velocity
           w (float): z-component of velocity
           mu (float): gravitational parameter
           iters (int): number of iterations of Newton-Raphson process
           atol (float): absolute tolerance of Newton-Raphson process

       Returns:
           float: alpha = 1/a
    """
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
        xi2 = xi**2
        z_i = alpha0*(xi2)
        a_i = (r0*vr0)/np.sqrt(mu)
        b_i = 1.0-alpha0*r0

        C_z_i = c2(z_i)
        S_z_i = c3(z_i)

        if np.isinf(C_z_i) == True or np.isinf(S_z_i) == True:
            return np.nan

        f_i = a_i*xi2*C_z_i + b_i*(xi**3)*S_z_i + r0*xi - np.sqrt(mu)*dt
        g_i = a_i*xi*(1.0-z_i*S_z_i) + b_i*xi2*C_z_i+r0
        ratio_i = f_i/g_i
        xi = xi - ratio_i
        i += 1

    return xi

def lagrangef_(xi, z, r):
    """Compute current value of Lagrange's f function.

       Args:
           xi (float): universal Kepler anomaly
           z (float): xi**2/alpha
           r (float): radial distance

       Returns:
           float: Lagrange's f function value
    """
    return 1.0-(xi**2)*c2(z)/r

def lagrangeg_(tau, xi, z, mu):
    """Compute current value of Lagrange's g function.

       Args:
           tau (float): time interval
           xi (float): universal Kepler anomaly
           z (float): xi**2/alpha
           r (float): radial distance

       Returns:
           float: Lagrange's g function value
    """
    return tau-(xi**3)*c3(z)/np.sqrt(mu)

def get_observations_data(mpc_object_data, inds):
    """Extract three ra/dec observations from MPC observation data file.

       Args:
           mpc_object_data (string): file path to MPC observation data of object
           inds (int array): indices of requested data

       Returns:
           obs_radec (1x3 SkyCoord array): ra/dec observation data
           obs_t (1x3 Time array): time observation data
           site_codes (1x3 int array): corresponding codes of observation sites
    """
    # construct SkyCoord 3-element array with observational information
    timeobs = np.zeros((3,), dtype=Time)
    obs_radec = np.zeros((3,), dtype=SkyCoord)
    obs_t = np.zeros((3,))

    timeobs[0] = Time( datetime(mpc_object_data['yr'][inds[0]],
                                mpc_object_data['month'][inds[0]],
                                mpc_object_data['day'][inds[0]]) + timedelta(days=mpc_object_data['utc'][inds[0]]) )
    timeobs[1] = Time( datetime(mpc_object_data['yr'][inds[1]],
                                mpc_object_data['month'][inds[1]],
                                mpc_object_data['day'][inds[1]]) + timedelta(days=mpc_object_data['utc'][inds[1]]) )
    timeobs[2] = Time( datetime(mpc_object_data['yr'][inds[2]],
                                mpc_object_data['month'][inds[2]],
                                mpc_object_data['day'][inds[2]]) + timedelta(days=mpc_object_data['utc'][inds[2]]) )

    obs_radec[0] = SkyCoord(mpc_object_data['radec'][inds[0]], unit=(uts.hourangle, uts.deg), obstime=timeobs[0])
    obs_radec[1] = SkyCoord(mpc_object_data['radec'][inds[1]], unit=(uts.hourangle, uts.deg), obstime=timeobs[1])
    obs_radec[2] = SkyCoord(mpc_object_data['radec'][inds[2]], unit=(uts.hourangle, uts.deg), obstime=timeobs[2])

    # construct vector of observation time (continous variable)
    obs_t[0] = obs_radec[0].obstime.tdb.jd
    obs_t[1] = obs_radec[1].obstime.tdb.jd
    obs_t[2] = obs_radec[2].obstime.tdb.jd

    site_codes = [mpc_object_data['observatory'][inds[0]],
                  mpc_object_data['observatory'][inds[1]],
                  mpc_object_data['observatory'][inds[2]]]

    return obs_radec, obs_t, site_codes

def get_observations_data_sat(iod_object_data, inds):
    """Extract three ra/dec observations from IOD observation data file.

       Args:
           iod_object_data (string): file path to sat tracking observation data of object
           inds (int array): indices of requested data

       Returns:
           obs_radec (1x3 SkyCoord array): ra/dec observation data
           obs_t (1x3 Time array): time observation data
           site_codes (1x3 int array): corresponding codes of observation sites
    """
    # construct SkyCoord 3-element array with observational information
    timeobs = np.zeros((3,), dtype=Time)
    obs_radec = np.zeros((3,), dtype=SkyCoord)
    obs_t = np.zeros((3,))

    td1 = timedelta(hours=1.0*iod_object_data['hr'][inds[0]],
                    minutes=1.0*iod_object_data['min'][inds[0]],
                    seconds=(iod_object_data['sec'][inds[0]]+iod_object_data['msec'][inds[0]]/1000.0))
    td2 = timedelta(hours=1.0*iod_object_data['hr'][inds[1]],
                    minutes=1.0*iod_object_data['min'][inds[1]],
                    seconds=(iod_object_data['sec'][inds[1]]+iod_object_data['msec'][inds[1]]/1000.0))
    td3 = timedelta(hours=1.0*iod_object_data['hr'][inds[2]],
                    minutes=1.0*iod_object_data['min'][inds[2]],
                    seconds=(iod_object_data['sec'][inds[2]]+iod_object_data['msec'][inds[2]]/1000.0))

    timeobs[0] = Time( datetime(iod_object_data['yr'][inds[0]],
                                iod_object_data['month'][inds[0]],
                                iod_object_data['day'][inds[0]]) + td1 )
    timeobs[1] = Time( datetime(iod_object_data['yr'][inds[1]],
                                iod_object_data['month'][inds[1]],
                                iod_object_data['day'][inds[1]]) + td2 )
    timeobs[2] = Time( datetime(iod_object_data['yr'][inds[2]],
                                iod_object_data['month'][inds[2]],
                                iod_object_data['day'][inds[2]]) + td3 )

    ra_ha0 = iod_object_data['right_ascension'][inds[0]] / 360.0 * 24.0
    ra_ha1 = iod_object_data['right_ascension'][inds[1]] / 360.0 * 24.0
    ra_ha2 = iod_object_data['right_ascension'][inds[2]] / 360.0 * 24.0

    dec0 = iod_object_data['declination'][inds[0]]
    dec1 = iod_object_data['declination'][inds[1]]
    dec2 = iod_object_data['declination'][inds[2]]

    obs_radec[0] = SkyCoord(ra=ra_ha0, dec=dec0, unit=(uts.hourangle, uts.deg), obstime=timeobs[0])
    obs_radec[1] = SkyCoord(ra=ra_ha1, dec=dec1, unit=(uts.hourangle, uts.deg), obstime=timeobs[1])
    obs_radec[2] = SkyCoord(ra=ra_ha2, dec=dec2, unit=(uts.hourangle, uts.deg), obstime=timeobs[2])

    # construct vector of observation time (continous variable)
    obs_t[0] = (timeobs[0]-timeobs[0]).sec
    obs_t[1] = (timeobs[1]-timeobs[0]).sec
    obs_t[2] = (timeobs[2]-timeobs[0]).sec

    site_codes = [iod_object_data['station'][inds[0]],
                  iod_object_data['station'][inds[1]],
                  iod_object_data['station'][inds[2]]]

    return obs_radec, obs_t, site_codes

def earth_ephemeris(t_tdb):
    """Compute heliocentric position of Earth at Julian date `t_tdb` (TDB, days),
    according to SPK kernel defined by astropy.coordinates.solar_system_ephemeris.

       Args:
           t_tdb (float): TDB instant of requested position

       Returns:
           (1x3 array): cartesian position in km
    """
    t = Time(t_tdb, format='jd', scale='tdb')
    ye = get_body_barycentric('earth', t)
    ys = get_body_barycentric('sun', t)
    y = ye - ys
    return y.xyz.value

def observer_wrt_sun(long, parallax_s, parallax_c, t_utc):
    """Compute position of observer at Earth's surface, with respect
    to the Sun, in equatorial frame.

       Args:
           long (float): longitude of observing site
           parallax_s (float): parallax constant S of observing site
           parallax_c (float): parallax constant C of observing site
           t_utc (Time): UTC time of observation

       Returns:
           (1x3 array): cartesian vector
    """
    t_jd_tdb = t_utc.tdb.jd
    xyz_es = earth_ephemeris(t_jd_tdb)
    xyz_oe = observerpos_mpc(long, parallax_s, parallax_c, t_utc)
    return (xyz_oe+xyz_es)/au

def object_wrt_sun(t_utc, a, e, taup, omega, I, Omega):
    """Compute position of celestial object with respect to the Sun, in equatorial frame.

       Args:
           t_utc (Time): UTC time of observation
           a (float): semimajor axis
           e (float): eccentricity
           taup (float): time of pericenter passage
           omega (float): argument of pericenter
           I (float): inclination
           Omega (float): longitude of ascending node

       Returns:
           (1x3 array): cartesian vector
    """
    t_jd_tdb = t_utc.tdb.jd
    xyz_eclip = orbel2xyz(t_jd_tdb, mu_Sun, a, e, taup, omega, I, Omega)
    return np.matmul(rot_eclip_to_equat, xyz_eclip)

def rho_vec(long, parallax_s, parallax_c, t_utc, a, e, taup, omega, I, Omega):
    """Compute slant range vector.

       Args:
           long (float): longitude of observing site
           parallax_s (float): parallax constant S of observing site
           parallax_c (float): parallax constant C of observing site
           t_utc (Time): UTC time of observation
           a (float): semimajor axis
           e (float): eccentricity
           taup (float): time of pericenter passage
           omega (float): argument of pericenter
           I (float): inclination
           Omega (float): longitude of ascending node

       Returns:
           (1x3 array): cartesian vector
    """
    return object_wrt_sun(t_utc, a, e, taup, omega, I, Omega)-observer_wrt_sun(long, parallax_s, parallax_c, t_utc)

def rhovec2radec(long, parallax_s, parallax_c, t_utc, a, e, taup, omega, I, Omega):
    """Transform slant range vector to ra/dec values.

       Args:
           long (float): longitude of observing site
           parallax_s (float): parallax constant S of observing site
           parallax_c (float): parallax constant C of observing site
           t_utc (Time): UTC time of observation
           a (float): semimajor axis
           e (float): eccentricity
           taup (float): time of pericenter passage
           omega (float): argument of pericenter
           I (float): inclination
           Omega (float): longitude of ascending node

       Returns:
           ra_rad (float): right ascension (rad)
           dec_rad (float): declination (rad)
    """
    r_v = rho_vec(long, parallax_s, parallax_c, t_utc, a, e, taup, omega, I, Omega)
    r_v_norm = np.linalg.norm(r_v, ord=2)
    r_v_unit = r_v/r_v_norm
    cosd_cosa = r_v_unit[0]
    cosd_sina = r_v_unit[1]
    sind = r_v_unit[2]
    ra_rad = np.arctan2(cosd_sina, cosd_cosa)
    dec_rad = np.arcsin(sind)
    if ra_rad <0.0:
        return ra_rad+2.0*np.pi, dec_rad
    else:
        return ra_rad, dec_rad

def angle_diff_rad(a1, a2):
    """Compute shortest signed difference between two angles. Input angles
    are assumed to be in radians. Result is returned in radians. Code adapted
    from https://rosettacode.org/wiki/Angle_difference_between_two_bearings#Python.

       Args:
            a1 (float): angle 1 in radians
            a2 (float): angle 2 in radians

       Returns:
           r (float): shortest signed difference in radians
    """
    r = (a2 - a1) % (2.0*np.pi)
    # Python modulus has same sign as divisor, which is positive here,
    # so no need to consider negative case
    if r >= np.pi:
        r -= (2.0*np.pi)
    return r

def radec_residual_mpc(x, t_ra_dec_datapoint, long, parallax_s, parallax_c):
    """Compute observed minus computed (O-C) residual for a given ra/dec
    datapoint, represented as a SkyCoord object, for MPC observation data.

       Args:
           x (1x6 array): set of Keplerian elements
           t_ra_dec_datapoint (SkyCoord): ra/dec datapoint
           long (float): longitude of observing site
           parallax_s (float): parallax constant S of observing site
           parallax_c (float): parallax constant C of observing site

       Returns:
           (1x2 array): right ascension difference, declination difference
    """
    ra_comp, dec_comp = rhovec2radec(long, parallax_s, parallax_c, t_ra_dec_datapoint.obstime,
                                     x[0], x[1], x[2], x[3], x[4], x[5])
    ra_obs, dec_obs = t_ra_dec_datapoint.ra.rad, t_ra_dec_datapoint.dec.rad
    #"unsigned" distance between points in torus
    diff_ra = angle_diff_rad(ra_obs, ra_comp)
    diff_dec = angle_diff_rad(dec_obs, dec_comp)
    return np.array((diff_ra,diff_dec))

def radec_residual_rov_mpc(x, t, ra_obs_rad, dec_obs_rad, long, parallax_s, parallax_c):
    """Compute right ascension and declination observed minus computed (O-C) residual,
    using precomputed vector of observed ra/dec values, for MPC observation data.

       Args:
           x (1x6 array): set of Keplerian elements
           t (Time): time of observation
           ra_obs_rad (float): observed right ascension (rad)
           dec_obs_rad (float): observed declination (rad)
           long (float): longitude of observing site
           parallax_s (float): parallax constant S of observing site
           parallax_c (float): parallax constant C of observing site

       Returns:
           (1x2 array): right ascension difference, declination difference
    """
    ra_comp, dec_comp = rhovec2radec(long, parallax_s, parallax_c, t, x[0], x[1], x[2], x[3], x[4], x[5])
    #"unsigned" distance between points in torus
    diff_ra = angle_diff_rad(ra_obs_rad, ra_comp)
    diff_dec = angle_diff_rad(dec_obs_rad, dec_comp)
    return np.array((diff_ra,diff_dec))

def get_observer_pos_wrt_sun(mpc_observatories_data, obs_radec, site_codes):
    """Compute position of observer at Earth's surface, with respect
    to the Sun, in equatorial frame, during 3 distinct instants.

       Args:
           mpc_observatories_data (string): path to file containing MPC observatories data.
           obs_radec (1x3 SkyCoord array): three rad/dec observations
           site_codes (1x3 int array): MPC codes of observation sites

       Returns:
           R (1x3 array): cartesian position vectors (observer wrt Sun)
           Ea_hc_pos (1x3 array): cartesian position vectors (Earth wrt Sun)
    """
    # astronomical unit in km
    Ea_hc_pos = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))
    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))
    # load MPC observatory data
    obsite1 = get_observatory_data(site_codes[0], mpc_observatories_data)
    obsite2 = get_observatory_data(site_codes[1], mpc_observatories_data)
    obsite3 = get_observatory_data(site_codes[2], mpc_observatories_data)
    # compute TDB instant of each observation
    t_jd1_tdb_val = obs_radec[0].obstime.tdb.jd
    t_jd2_tdb_val = obs_radec[1].obstime.tdb.jd
    t_jd3_tdb_val = obs_radec[2].obstime.tdb.jd

    Ea_jd1 = earth_ephemeris(t_jd1_tdb_val)
    Ea_jd2 = earth_ephemeris(t_jd2_tdb_val)
    Ea_jd3 = earth_ephemeris(t_jd3_tdb_val)

    Ea_hc_pos[0] = Ea_jd1/au
    Ea_hc_pos[1] = Ea_jd2/au
    Ea_hc_pos[2] = Ea_jd3/au

    R[0] = (  Ea_jd1 + observerpos_mpc(obsite1['Long'], obsite1['sin'], obsite1['cos'], obs_radec[0].obstime)  )/au
    R[1] = (  Ea_jd2 + observerpos_mpc(obsite2['Long'], obsite2['sin'], obsite2['cos'], obs_radec[1].obstime)  )/au
    R[2] = (  Ea_jd3 + observerpos_mpc(obsite3['Long'], obsite3['sin'], obsite3['cos'], obs_radec[2].obstime)  )/au

    return R, Ea_hc_pos

def get_observer_pos_wrt_earth(sat_observatories_data, obs_radec, site_codes):
    """Compute position of observer at Earth's surface, with respect
    to the Earth, in equatorial frame, during 3 distinct instants.

       Args:
           sat_observatories_data (string): path to file containing COSPAR satellite tracking stations data.
           obs_radec (1x3 SkyCoord array): three rad/dec observations
           site_codes (1x3 int array): COSPAR codes of observation sites

       Returns:
           R (1x3 array): cartesian position vectors (observer wrt Earth)
    """
    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))
    # load MPC observatory data
    obsite1 = get_station_data(site_codes[0], sat_observatories_data)
    obsite2 = get_station_data(site_codes[1], sat_observatories_data)
    obsite3 = get_station_data(site_codes[2], sat_observatories_data)

    R[0] = observerpos_sat(obsite1['Latitude'], obsite1['Longitude'], obsite1['Elev'], obs_radec[0].obstime)
    R[1] = observerpos_sat(obsite2['Latitude'], obsite2['Longitude'], obsite2['Elev'], obs_radec[1].obstime)
    R[2] = observerpos_sat(obsite3['Latitude'], obsite3['Longitude'], obsite3['Elev'], obs_radec[2].obstime)

    return R

def gauss_method_core(obs_radec, obs_t, R, mu, r2_root_ind=0):
    """Perform core Gauss method.

       Args:
           obs_radec (1x3 SkyCoord array): three rad/dec observations
           obs_t (1x3 array): three times of observations
           R (1x3 array): three observer position vectors
           mu (float): gravitational parameter of center of attraction
           r2_root_ind (int): index of Gauss polynomial root

       Returns:
           r1 (1x3 array): estimated position at first observation
           r2 (1x3 array): estimated position at second observation
           r3 (1x3 array): estimated position at third observation
           v2 (1x3 array): estimated velocity at second observation
           D (3x3 array): auxiliary matrix
           rho1 (1x3 array): LOS vector at first observation
           rho2 (1x3 array): LOS vector at second observation
           rho3 (1x3 array): LOS vector at third observation
           tau1 (float): time interval from second to first observation
           tau3 (float): time interval from second to third observation
           f1 (float): estimated Lagrange's f function value at first observation
           g1 (float): estimated Lagrange's g function value at first observation
           f3 (float): estimated Lagrange's f function value at third observation
           g3 (float): estimated Lagrange's g function value at third observation
           rho_1_sr (float): estimated slant range at first observation
           rho_2_sr (float): estimated slant range at second observation
           rho_3_sr (float): estimated slant range at third observation
    """
    # get Julian date of observations
    t1 = obs_t[0]
    t2 = obs_t[1]
    t3 = obs_t[2]

    # compute Line-Of-Sight (LOS) vectors
    rho1 = losvector(obs_radec[0].ra.rad, obs_radec[0].dec.rad)
    rho2 = losvector(obs_radec[1].ra.rad, obs_radec[1].dec.rad)
    rho3 = losvector(obs_radec[2].ra.rad, obs_radec[2].dec.rad)

    # compute time differences; make sure time units are consistent!
    tau1 = (t1-t2)
    tau3 = (t3-t2)
    tau = (tau3-tau1)

    p = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    p[0] = np.cross(rho2, rho3)
    p[1] = np.cross(rho1, rho3)
    p[2] = np.cross(rho1, rho2)

    D0  = np.dot(rho1, p[0])

    D = np.zeros((3,3))

    for i in range(0,3):
        for j in range(0,3):
            D[i,j] = np.dot(R[i], p[j])

    A = (-D[0,1]*(tau3/tau)+D[1,1]+D[2,1]*(tau1/tau))/D0
    B = (D[0,1]*(tau3**2-tau**2)*(tau3/tau)+D[2,1]*(tau**2-tau1**2)*(tau1/tau))/(6*D0)

    E = np.dot(R[1], rho2)
    Rsub2p2 = np.dot(R[1], R[1])

    a = -(A**2+2.0*A*E+Rsub2p2)
    b = -2.0*mu*B*(A+E)
    c = -(mu**2)*(B**2)

    #get all real, positive solutions to the Gauss polynomial
    gauss_poly_coeffs = np.zeros((9,))
    gauss_poly_coeffs[0] = 1.0
    gauss_poly_coeffs[2] = a
    gauss_poly_coeffs[5] = b
    gauss_poly_coeffs[8] = c

    gauss_poly_roots = np.roots(gauss_poly_coeffs)
    rt_indx = np.where( np.isreal(gauss_poly_roots) & (gauss_poly_roots >= 0.0) )
    if len(rt_indx[0]) > 1: #-1:#
        print('WARNING: Gauss polynomial has more than 1 real, positive solution')
        print('gauss_poly_coeffs = ', gauss_poly_coeffs)
        print('gauss_poly_roots = ', gauss_poly_roots)
        print('len(rt_indx[0]) = ', len(rt_indx[0]))
        print('np.real(gauss_poly_roots[rt_indx[0]]) = ', np.real(gauss_poly_roots[rt_indx[0]]))
        print('r2_root_ind = ', r2_root_ind)

    r2_star = np.real(gauss_poly_roots[rt_indx[0][r2_root_ind]])
    print('r2_star = ', r2_star)

    num1 = 6.0*(D[2,0]*(tau1/tau3)+D[1,0]*(tau/tau3))*(r2_star**3)+mu*D[2,0]*(tau**2-tau1**2)*(tau1/tau3)
    den1 = 6.0*(r2_star**3)+mu*(tau**2-tau3**2)

    rho_1_sr = ((num1/den1)-D[0,0])/D0

    rho_2_sr = A+(mu*B)/(r2_star**3)

    num3 = 6.0*(D[0,2]*(tau3/tau1)-D[1,2]*(tau/tau1))*(r2_star**3)+mu*D[0,2]*(tau**2-tau3**2)*(tau3/tau1)
    den3 = 6.0*(r2_star**3)+mu*(tau**2-tau1**2)

    rho_3_sr = ((num3/den3)-D[2,2])/D0

    r1 = R[0]+rho_1_sr*rho1
    r2 = R[1]+rho_2_sr*rho2
    r3 = R[2]+rho_3_sr*rho3

    f1 = lagrangef(mu, r2_star, tau1)
    f3 = lagrangef(mu, r2_star, tau3)

    g1 = lagrangeg(mu, r2_star, tau1)
    g3 = lagrangeg(mu, r2_star, tau3)

    v2 = (-f3*r1+f1*r3)/(f1*g3-f3*g1)

    return r1, r2, r3, v2, D, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_sr, rho_2_sr, rho_3_sr

def gauss_refinement(mu, tau1, tau3, r2, v2, atol, D, R, rho1, rho2, rho3, f_1, g_1, f_3, g_3):
    """Perform refinement of Gauss method.

       Args:
           mu (float): gravitational parameter of center of attraction
           tau1 (float): time interval from second to first observation
           tau3 (float): time interval from second to third observation
           r2 (1x3 array): estimated position at second observation
           v2 (1x3 array): estimated velocity at second observation
           atol (float): absolute tolerance of universal Kepler anomaly computation
           D (3x3 array): auxiliary matrix
           R (1x3 array): three observer position vectors
           rho1 (1x3 array): LOS vector at first observation
           rho2 (1x3 array): LOS vector at second observation
           rho3 (1x3 array): LOS vector at third observation
           f_1 (float): estimated Lagrange's f function value at first observation
           g_1 (float): estimated Lagrange's g function value at first observation
           f_3 (float): estimated Lagrange's f function value at third observation
           g_3 (float): estimated Lagrange's g function value at third observation

       Returns:
           r1 (1x3 array): updated position at first observation
           r2 (1x3 array): updated position at second observation
           r3 (1x3 array): updated position at third observation
           v2 (1x3 array): updated velocity at second observation
           rho_1_sr (float): updated slant range at first observation
           rho_2_sr (float): updated slant range at second observation
           rho_3_sr (float): updated slant range at third observation
           f_1_new (float): updated Lagrange's f function value at first observation
           g_1_new (float): updated Lagrange's g function value at first observation
           f_3_new (float): updated Lagrange's f function value at third observation
           g_3_new (float): updated Lagrange's g function value at third observation
    """
    refinement_success = 1

    xi1 = univkepler(tau1, r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu, iters=10, atol=atol)
    xi3 = univkepler(tau3, r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu, iters=10, atol=atol)

    if np.isnan(xi1) == True or np.isnan(xi3) == True:
        refinement_success = 0
        return np.nan, r2, np.nan, v2, rho1, rho2, rho3, f_1, g_1, f_3, g_3, refinement_success

    r0_ = np.sqrt((r2[0]**2)+(r2[1]**2)+(r2[2]**2))
    v20_ = (v2[0]**2)+(v2[1]**2)+(v2[2]**2)
    alpha0_ = (2.0/r0_)-(v20_/mu)

    z1_ = alpha0_*(xi1**2)
    f_1_new = (f_1+lagrangef_(xi1, z1_, r0_))/2
    g_1_new = (g_1+lagrangeg_(tau1, xi1, z1_, mu))/2

    z3_ = alpha0_*(xi3**2)
    f_3_new = (f_3+lagrangef_(xi3, z3_, r0_))/2
    g_3_new = (g_3+lagrangeg_(tau3, xi3, z3_, mu))/2

    denum = f_1_new*g_3_new-f_3_new*g_1_new
    if np.isinf(np.abs(denum)) == True:
        # one of the terms in denum became really big :(
        refinement_success = 0
        return np.nan, r2, np.nan, v2, rho1, rho2, rho3, f_1, g_1, f_3, g_3, refinement_success

    c1_ = g_3_new/denum
    c3_ = -g_1_new/denum

    D0  = np.dot(rho1, np.cross(rho2, rho3))

    rho_1_sr = (-D[0,0]+D[1,0]/c1_-D[2,0]*(c3_/c1_))/D0
    rho_2_sr = (-c1_*D[0,1]+D[1,1]-c3_*D[2,1])/D0
    rho_3_sr = (-D[0,2]*(c1_/c3_)+D[1,2]/c3_-D[2,2])/D0

    r1 = R[0]+rho_1_sr*rho1
    r2 = R[1]+rho_2_sr*rho2
    r3 = R[2]+rho_3_sr*rho3

    v2 = (-f_3_new*r1+f_1_new*r3)/denum

    return r1, r2, r3, v2, rho_1_sr, rho_2_sr, rho_3_sr, f_1_new, g_1_new, f_3_new, g_3_new, refinement_success

def gauss_estimate_mpc(mpc_object_data, mpc_observatories_data, inds, r2_root_ind=0):
    """Gauss method implementation for MPC Near-Earth asteroids ra/dec tracking data.

       Args:
           mpc_object_data (string): path to MPC-formatted observation data file
           mpc_observatories_data (string): path to MPC observation sites data file
           inds (1x3 int array): indices of requested data
           r2_root_ind (int): index of selected Gauss polynomial root

       Returns:
           r1 (1x3 array): updated position at first observation
           r2 (1x3 array): updated position at second observation
           r3 (1x3 array): updated position at third observation
           v2 (1x3 array): updated velocity at second observation
           D (3x3 array): auxiliary matrix
           R (1x3 array): three observer position vectors
           rho1 (1x3 array): LOS vector at first observation
           rho2 (1x3 array): LOS vector at second observation
           rho3 (1x3 array): LOS vector at third observation
           tau1 (float): time interval from second to first observation
           tau3 (float): time interval from second to third observation
           f1 (float): Lagrange's f function value at first observation
           g1 (float): Lagrange's g function value at first observation
           f3 (float): Lagrange's f function value at third observation
           g3 (float): Lagrange's g function value at third observation
           Ea_hc_pos (1x3 array): cartesian position vectors (Earth wrt Sun)
           rho_1_sr (float): slant range at first observation
           rho_2_sr (float): slant range at second observation
           rho_3_sr (float): slant range at third observation
           obs_t (1x3 array): three times of observations
    """
    # mu_Sun = 0.295912208285591100E-03 # Sun's G*m, au^3/day^2
    mu = mu_Sun # cts.GM_sun.to(uts.Unit("au3 / day2")).value

    # extract observations data
    obs_radec, obs_t, site_codes = get_observations_data(mpc_object_data, inds)

    # compute observer position vectors wrt Sun
    R, Ea_hc_pos = get_observer_pos_wrt_sun(mpc_observatories_data, obs_radec, site_codes)

    # perform core Gauss method
    r1, r2, r3, v2, D, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_sr, rho_2_sr, rho_3_sr = \
        gauss_method_core(obs_radec, obs_t, R, mu, r2_root_ind=r2_root_ind)

    return r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3,\
           f1, g1, f3, g3, Ea_hc_pos, rho_1_sr, rho_2_sr, rho_3_sr, obs_t

# Implementation of Gauss method for IOD-formatted optical observations of Earth satellites
def gauss_estimate_sat(iod_object_data, sat_observatories_data, inds, r2_root_ind=0):
    """Gauss method implementation for Earth-orbiting satellites ra/dec tracking data.
    Assumes observation data uses IOD format, with angle subformat 2.

       Args:
           iod_object_data (string): file path to sat tracking observation data of object
           sat_observatories_data (string): path to file containing COSPAR satellite tracking stations data.
           inds (1x3 int array): line numbers in data file to be processed
           r2_root_ind (int): index of selected Gauss polynomial root

       Returns:
           r1 (1x3 array): updated position at first observation
           r2 (1x3 array): updated position at second observation
           r3 (1x3 array): updated position at third observation
           v2 (1x3 array): updated velocity at second observation
           D (3x3 array): auxiliary matrix
           R (1x3 array): three observer position vectors
           rho1 (1x3 array): LOS vector at first observation
           rho2 (1x3 array): LOS vector at second observation
           rho3 (1x3 array): LOS vector at third observation
           tau1 (float): time interval from second to first observation
           tau3 (float): time interval from second to third observation
           f1 (float): Lagrange's f function value at first observation
           g1 (float): Lagrange's g function value at first observation
           f3 (float): Lagrange's f function value at third observation
           g3 (float): Lagrange's g function value at third observation
           rho_1_sr (float): slant range at first observation
           rho_2_sr (float): slant range at second observation
           rho_3_sr (float): slant range at third observation
           obs_t_jd (1x3 array): three Julian dates of observations
    """
    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    mu = mu_Earth

    # extract observations data
    obs_radec, obs_t, site_codes = get_observations_data_sat(iod_object_data, inds)
    obs_t_jd = np.array((obs_radec[0].obstime.jd, obs_radec[1].obstime.jd, obs_radec[2].obstime.jd))

    # compute observer position vectors wrt Sun
    R = get_observer_pos_wrt_earth(sat_observatories_data, obs_radec, site_codes)

    # perform core Gauss method
    r1, r2, r3, v2, D, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_sr, rho_2_sr, rho_3_sr = \
        gauss_method_core(obs_radec, obs_t, R, mu, r2_root_ind=r2_root_ind)

    return r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_sr, rho_2_sr, rho_3_sr, obs_t_jd

def gauss_iterator_sat(iod_object_data, sat_observatories_data, inds, refiters=0, r2_root_ind=0):
    """Gauss method iterator for Earth-orbiting satellites ra/dec tracking data.
    Computes a first estimate of the orbit using gauss_estimate_sat function, and
    then refines this estimate using gauss_refinement. Assumes observation data
    file is IOD-formatted, with angle subformat 2.

       Args:
           iod_object_data (string): file path to sat tracking observation data of object
           sat_observatories_data (string): path to file containing COSPAR satellite tracking stations data.
           inds (1x3 int array): line numbers in data file to be processed
           refiters (int): number of refinement iterations to be performed
           r2_root_ind (int): index of selected Gauss polynomial root

       Returns:
           r1 (1x3 array): updated position at first observation
           r2 (1x3 array): updated position at second observation
           r3 (1x3 array): updated position at third observation
           v2 (1x3 array): updated velocity at second observation
           R (1x3 array): three observer position vectors
           rho1 (1x3 array): LOS vector at first observation
           rho2 (1x3 array): LOS vector at second observation
           rho3 (1x3 array): LOS vector at third observation
           rho_1_sr (float): slant range at first observation
           rho_2_sr (float): slant range at second observation
           rho_3_sr (float): slant range at third observation
           obs_t (1x3 array): times of observations
    """
    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    mu = mu_Earth
    r1_est, r2_est, r3_est, v2_est, D_est, R_est, rho1_est, rho2_est, rho3_est, tau1_est, tau3_est, f1_est, g1_est, f3_est, g3_est, rho_1_sr_est, rho_2_sr_est, rho_3_sr_est, obs_t_est = \
        gauss_estimate_sat(iod_object_data, sat_observatories_data, inds, r2_root_ind=r2_root_ind)

    r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_sr, rho_2_sr, rho_3_sr, obs_t = r1_est, r2_est, r3_est, v2_est, D_est, R_est, rho1_est, rho2_est, rho3_est, tau1_est, tau3_est, f1_est, g1_est, f3_est, g3_est, rho_1_sr_est, rho_2_sr_est, rho_3_sr_est, obs_t_est


    # Apply refinement to Gauss' method, `refiters` iterations
    refinement_success = 0

    for i in range(0,refiters):
        r1, r2, r3, v2, rho_1_sr, rho_2_sr, rho_3_sr, f1, g1, f3, g3, refinement_success = \
            gauss_refinement(mu, tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)


    if refinement_success == 1:
        # refinement worked
        return r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_sr, rho_2_sr, rho_3_sr, obs_t, refinement_success

    else:
        return r1_est, r2_est, r3_est, v2_est, R_est, rho1_est, rho2_est, rho3_est, rho_1_sr_est, rho_2_sr_est, rho_3_sr_est, obs_t_est, refinement_success

def gauss_iterator_mpc(mpc_object_data, mpc_observatories_data, inds, refiters=0, r2_root_ind=0):
    """Gauss method iterator for minor planets ra/dec tracking data.
    Computes a first estimate of the orbit using gauss_estimate_sat function, and
    then refines this estimate using gauss_refinement. Assumes observation data
    file follows MPC format.

       Args:
           mpc_object_data (string): path to MPC-formatted observation data file
           mpc_observatories_data (string): path to MPC observation sites data file
           inds (1x3 int array): line numbers in data file to be processed
           refiters (int): number of refinement iterations to be performed
           r2_root_ind (int): index of selected Gauss polynomial root

       Returns:
           r1 (1x3 array): updated position at first observation
           r2 (1x3 array): updated position at second observation
           r3 (1x3 array): updated position at third observation
           v2 (1x3 array): updated velocity at second observation
           R (1x3 array): three observer position vectors
           rho1 (1x3 array): LOS vector at first observation
           rho2 (1x3 array): LOS vector at second observation
           rho3 (1x3 array): LOS vector at third observation
           rho_1_sr (float): slant range at first observation
           rho_2_sr (float): slant range at second observation
           rho_3_sr (float): slant range at third observation
           Ea_hc_pos (1x3 array): cartesian position vectors (Earth wrt Sun)
           obs_t (1x3 array): times of observations
    """
    # mu_Sun = 0.295912208285591100E-03 # Sun's G*m, au^3/day^2
    mu = mu_Sun # cts.GM_sun.to(uts.Unit("au3 / day2")).value
    r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos, rho_1_sr, rho_2_sr, rho_3_sr, obs_t =\
        gauss_estimate_mpc(mpc_object_data, mpc_observatories_data, inds, r2_root_ind=r2_root_ind)

    # Apply refinement to Gauss' method, `refiters` iterations
    for i in range(0,refiters):
        r1, r2, r3, v2, rho_1_sr, rho_2_sr, rho_3_sr, f1, g1, f3, g3, refinement_success= \
            gauss_refinement(mu, tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)

    return r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_sr, rho_2_sr, rho_3_sr, Ea_hc_pos, obs_t

def radec_obs_vec_sat(inds, iod_object_data):
    """Compute vector of observed ra,dec values for satellite tracking data (IOD-formatted).

       Args:
           inds (int array): line numbers of data in file
           iod_object_data (ndarray): observation data

       Returns:
           rov (1xlen(inds) array): vector of ra/dec observed values
    """
    rov = np.zeros((2*len(inds)))
    for i in range(len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1],
                       minutes=1.0*iod_object_data['min'][indm1],
                       seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1],
                                 iod_object_data['month'][indm1],
                                 iod_object_data['day'][indm1]) + td )

        ra_ha = iod_object_data['right_ascension'][indm1] / 360.0 * 24.0
        dec = iod_object_data['declination'][indm1]

        obs_t_ra_dec = SkyCoord(ra=ra_ha, dec=dec, unit=(uts.hourangle, uts.deg), obstime=timeobs)
        rov[2*i-2], rov[2*i-1] = obs_t_ra_dec.ra.rad, obs_t_ra_dec.dec.rad
    return rov

def radec_res_vec_rov_sat(x, inds, iod_object_data, sat_observatories_data, rov):
    """Compute vector of observed minus computed (O-C) residuals for ra/dec Earth-orbiting satellite observations
    with pre-computed observed radec values vector. Assumes ra/dec observed values vector
    is contained in rov, and they are stored as rov = [ra1, dec1, ra2, dec2, ...].

       Args:
           x (1x6 float array): set of orbital elements (a, e, taup, omega, I, Omega)
           inds (int array): line numbers of data in file
           iod_object_data (ndarray): observation data
           sat_observatories_data (ndarray): satellite tracking stations data
           rov (1xlen(inds) float-like array): vector of observed ra/dec values

       Returns:
           rv (1xlen(inds) array): vector of ra/dec (O-C) residuals.
    """
    rv = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1],
                       minutes=1.0*iod_object_data['min'][indm1],
                       seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1],
                                 iod_object_data['month'][indm1],
                                 iod_object_data['day'][indm1]) + td )
        site_code = iod_object_data['station'][indm1]
        obsite = get_station_data(site_code, sat_observatories_data)
        # object position wrt to Earth
        xyz_obj = orbel2xyz(timeobs.jd,
                            cts.GM_earth.to(uts.Unit('km3 / day2')).value,
                            x[0], x[1], x[2], x[3], x[4], x[5])
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
        # store O-C residual into vector (O-C = "Observed minus Computed")
        rv[2*i-2], rv[2*i-1] = diff_ra, diff_dec
    return rv

# compute residuals vector for ra/dec observations; return observation times and residual vector
# inds = obs_arr
def t_radec_res_vec_sat(x, inds, iod_object_data, sat_observatories_data, rov):
    """Compute vector of observed minus computed (O-C) residuals for ra/dec Earth-orbiting satellite observations
    with pre-computed observed radec values vector. Assumes ra/dec observed values vector
    is contained in rov, and they are stored as rov = [ra1, dec1, ra2, dec2, ...].

       Args:
           x (1x6 float array): set of orbital elements (a, e, taup, omega, I, Omega)
           inds (int array): line numbers of data in file
           iod_object_data (ndarray): observation data
           sat_observatories_data (ndarray): satellite tracking stations data
           rov (1xlen(inds) float-like array): vector of observed ra/dec values

       Returns:
           rv (1xlen(inds) array): vector of ra/dec (O-C) residuals.
           tv (1xlen(inds) array): vector of observation times.
    """
    rv = np.zeros((2*len(inds)))
    tv = np.zeros((len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1],
                       minutes=1.0*iod_object_data['min'][indm1],
                       seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1],
                                 iod_object_data['month'][indm1],
                                 iod_object_data['day'][indm1]) + td )
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
        # store O-C residual into vector (O-C = "Observed minus Computed")
        rv[2*i-2], rv[2*i-1] = diff_ra, diff_dec
        tv[i] = t_jd
    return tv, rv

# compute auxiliary vector of observed ra,dec values
def radec_obs_vec_mpc(inds, mpc_object_data):
    """Compute vector of observed ra,dec values for MPC tracking data.

       Args:
           inds (int array): line numbers of data in file
           mpc_object_data (ndarray): MPC observation data for object

       Returns:
           rov (1xlen(inds) array): vector of ra/dec observed values
    """
    rov = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        timeobs = Time( datetime(mpc_object_data['yr'][indm1],
                                 mpc_object_data['month'][indm1],
                                 mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
        obs_t_ra_dec = SkyCoord(mpc_object_data['radec'][indm1], unit=(uts.hourangle, uts.deg), obstime=timeobs)
        rov[2*i-2], rov[2*i-1] = obs_t_ra_dec.ra.rad, obs_t_ra_dec.dec.rad
    return rov

# compute residuals vector for ra/dec observations with pre-computed observed radec values vector
def radec_res_vec_rov_mpc(x, inds, mpc_object_data, mpc_observatories_data, rov):
    """Compute vector of observed minus computed (O-C) residuals for ra/dec
    MPC-formatted observations of minor planets (asteroids, comets, etc.), with
    pre-computed observed radec values vector. Assumes ra/dec observed values
    vector is contained in rov, and they are stored as
    rov = [ra1, dec1, ra2, dec2, ...].

       Args:
           x (1x6 float array): set of orbital elements (a, e, taup, omega, I, Omega)
           inds (int array): line numbers of data in file
           mpc_object_data (ndarray): observation data
           mpc_observatories_data (ndarray): MPC observatories data
           rov (1xlen(inds) float-like array): vector of observed ra/dec values

       Returns:
           rv (1xlen(inds) array): vector of ra/dec (O-C) residuals.
    """
    rv = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        timeobs = Time( datetime(mpc_object_data['yr'][indm1],
                                 mpc_object_data['month'][indm1],
                                 mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
        site_code = mpc_object_data['observatory'][indm1]
        obsite = get_observatory_data(site_code, mpc_observatories_data)
        # compute residuals
        radec_res = radec_residual_rov_mpc(x, timeobs, rov[2*i-2], rov[2*i-1], obsite['Long'], obsite['sin'], obsite['cos'])
        # assign residuals to ra/dec residuals vector
        rv[2*i-2], rv[2*i-1] = radec_res
    return rv

# compute residuals vector for ra/dec observations with pre-computed observed radec values vector;
# return observation times and residual vector
def t_radec_res_vec_mpc(x, inds, mpc_object_data, mpc_observatories_data):
    """Compute vector of observed minus computed (O-C) residuals for ra/dec
    MPC-formatted observations of minor planets (asteroids, comets, etc.), with
    pre-computed observed radec values vector. Assumes ra/dec observed values
    vector is contained in rov, and they are stored as
    rov = [ra1, dec1, ra2, dec2, ...].

       Args:
           x (1x6 float array): set of orbital elements (a, e, taup, omega, I, Omega)
           inds (int array): line numbers of data in file
           mpc_object_data (ndarray): observation data
           mpc_observatories_data (ndarray): MPC observatories data
           rov (1xlen(inds) float-like array): vector of observed ra/dec values

       Returns:
           rv (1xlen(inds) array): vector of ra/dec (O-C) residuals.
           tv (1xlen(inds) array): vector of observation times.
    """
    rv = np.zeros((2*len(inds)))
    tv = np.zeros((len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        timeobs = Time( datetime(mpc_object_data['yr'][indm1],
                                 mpc_object_data['month'][indm1],
                                 mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
        site_code = mpc_object_data['observatory'][indm1]
        obs_t_ra_dec = SkyCoord(mpc_object_data['radec'][indm1], unit=(uts.hourangle, uts.deg), obstime=timeobs)
        obsite = get_observatory_data(site_code, mpc_observatories_data)
        # compute residuals
        radec_res = radec_residual_mpc(x, obs_t_ra_dec, obsite['Long'], obsite['sin'], obsite['cos'])
        rv[2*i-2], rv[2*i-1] = radec_res
        # assign residuals to rd/dec residuals vector
        tv[i] = timeobs.tdb.jd
    return tv, rv

def gauss_method_mpc(filename, bodyname, obs_arr, r2_root_ind_vec=None, refiters=0, plot=True):
    """Gauss method high-level function for minor planets (asteroids, comets,
    etc.) orbit determination from MPC-formatted ra/dec tracking data. Roots of
    8-th order Gauss polynomial are computed using np.roots function. Note that
    if `r2_root_ind_vec` is not specified by the user, then the first positive
    root returned by np.roots is used by default.

       Args:
           filename (string): path to MPC-formatted observation data file
           bodyname (string): user-defined name of minor planet
           obs_arr (int vector): line numbers in data file to be processed
           refiters (int): number of refinement iterations to be performed
           r2_root_ind_vec (1xlen(obs_arr) int array): indices of Gauss polynomial roots.
           plot (bool): if True, plots data.

       Returns:
           x (tuple): set of Keplerian orbital elements (a, e, taup, omega, I, omega, T)
    """
    # load MPC data for a given NEA
    mpc_object_data = load_mpc_data(filename)

    #load MPC data of listed observatories (longitude, parallax constants C, S)
    mpc_observatories_data = load_mpc_observatories_data('mpc_observatories.txt')

    #definition of the astronomical unit in km
    # au = cts.au.to(uts.Unit('km')).value

    # Sun's G*m value
    # mu_Sun = 0.295912208285591100E-03 # au^3/day^2
    mu = mu_Sun # cts.GM_sun.to(uts.Unit("au3 / day2")).value

    #the total number of observations used
    nobs = len(obs_arr)

    # if r2_root_ind_vec was not specified, then use always the first positive root by default
    if r2_root_ind_vec is None:
        r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)

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
    x_Ea_vec = np.zeros((nobs,))
    y_Ea_vec = np.zeros((nobs,))
    z_Ea_vec = np.zeros((nobs,))
    t_vec = np.zeros((nobs,))

    for j in range (0,nobs-2):
        # Apply Gauss method to three elements of data
        inds = [obs_arr[j]-1, obs_arr[j+1]-1, obs_arr[j+2]-1]
        print('Processing observation #', j)
        r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_sr, rho_2_sr, rho_3_sr, Ea_hc_pos, obs_t = \
            gauss_iterator_mpc(mpc_object_data, mpc_observatories_data, inds, refiters=refiters, r2_root_ind=r2_root_ind_vec[j])

        if j==0:
            t_vec[0] = obs_t[0]
            x_vec[0], y_vec[0], z_vec[0] = np.matmul(rot_equat_to_eclip, r1)
            x_Ea_vec[0], y_Ea_vec[0], z_Ea_vec[0] = np.matmul(rot_equat_to_eclip, earth_ephemeris(obs_t[0])/au)
        if j==nobs-3:
            t_vec[nobs-1] = obs_t[2]
            x_vec[nobs-1], y_vec[nobs-1], z_vec[nobs-1] = np.matmul(rot_equat_to_eclip, r3)
            x_Ea_vec[nobs-1], y_Ea_vec[nobs-1], z_Ea_vec[nobs-1] = np.matmul(rot_equat_to_eclip, earth_ephemeris(obs_t[2])/au)

        r2_eclip = np.matmul(rot_equat_to_eclip, r2)
        v2_eclip = np.matmul(rot_equat_to_eclip, v2)

        a_num = semimajoraxis(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2], mu)
        e_num = eccentricity(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2], mu)
        f_num = trueanomaly5(r2_eclip[0], r2_eclip[1], r2_eclip[2], v2_eclip[0], v2_eclip[1], v2_eclip[2], mu)
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
        x_Ea_vec[j+1] = Ea_hc_pos_eclip[0]
        y_Ea_vec[j+1] = Ea_hc_pos_eclip[1]
        z_Ea_vec[j+1] = Ea_hc_pos_eclip[2]

    a_mean = np.mean(a_vec) #au
    e_mean = np.mean(e_vec) #dimensionless
    taup_mean = np.mean(taup_vec) #deg
    w_mean = np.mean(w_vec) #deg
    I_mean = np.mean(I_vec) #deg
    W_mean = np.mean(W_vec) #deg
    n_mean = np.mean(n_vec) #sec

    print('\n*** ORBIT DETERMINATION: GAUSS METHOD ***')
    print('Observational arc:')
    print('Number of observations: ', len(obs_arr))
    print('First observation (UTC) : ', Time(t_vec[0], format='jd').iso)
    print('Last observation (UTC) : ', Time(t_vec[-1], format='jd').iso)

    print('\nAVERAGE ORBITAL ELEMENTS (ECLIPTIC, MEAN J2000.0): a, e, taup, omega, I, Omega, T')
    print('Semi-major axis (a):                 ', a_mean, 'au')
    print('Eccentricity (e):                    ', e_mean)
    print('Time of pericenter passage (tau):    ', Time(taup_mean, format='jd').iso, 'JDTDB')
    print('Pericenter distance (q):             ', a_mean*(1.0-e_mean), 'au')
    print('Apocenter distance (Q):              ', a_mean*(1.0+e_mean), 'au')
    print('Argument of pericenter (omega):      ', w_mean, 'deg')
    print('Inclination (I):                     ', I_mean, 'deg')
    print('Longitude of Ascending Node (Omega): ', W_mean, 'deg')
    print('Orbital period (T):                  ', 2.0*np.pi/n_mean, 'days')

    # PLOT
    if plot:
        npoints = 500 # number of points in orbit
        theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
        t_Ea_vec = np.linspace(t_vec[0], t_vec[-1], npoints)
        x_orb_vec = np.zeros((npoints,))
        y_orb_vec = np.zeros((npoints,))
        z_orb_vec = np.zeros((npoints,))
        x_Ea_orb_vec = np.zeros((npoints,))
        y_Ea_orb_vec = np.zeros((npoints,))
        z_Ea_orb_vec = np.zeros((npoints,))

        for i in range(0,npoints):
            x_orb_vec[i], y_orb_vec[i], z_orb_vec[i] = xyz_frame2(a_mean, e_mean, theta_vec[i],
                                                                  np.deg2rad(w_mean), np.deg2rad(I_mean), np.deg2rad(W_mean))
            xyz_Ea_orb_vec_equat = earth_ephemeris(t_Ea_vec[i])/au
            xyz_Ea_orb_vec_eclip = np.matmul(rot_equat_to_eclip, xyz_Ea_orb_vec_equat)
            x_Ea_orb_vec[i], y_Ea_orb_vec[i], z_Ea_orb_vec[i] = xyz_Ea_orb_vec_eclip

        ax = plt.axes(aspect='equal', projection='3d')

        # Sun-centered orbits: Computed orbit and Earth's
        ax.scatter3D(0.0, 0.0, 0.0, color='yellow', label='Sun')
        ax.scatter3D(x_Ea_vec, y_Ea_vec, z_Ea_vec, color='blue', marker='.', label='Earth orbit')
        ax.plot3D(x_Ea_orb_vec, y_Ea_orb_vec, z_Ea_orb_vec, color='blue', linewidth=0.5)
        ax.scatter3D(x_vec, y_vec, z_vec, color='red', marker='+', label=bodyname+' orbit')
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
        ax.set_title('Angles-only orbit determ. (Gauss): '+bodyname)
        plt.show()

    return a_mean, e_mean, taup_mean, w_mean, I_mean, W_mean, 2.0*np.pi/n_mean

def gauss_method_sat(filename, obs_arr=None, bodyname=None, r2_root_ind_vec=None, refiters=10, plot=False, mode_of_observationsequence = 0):
    """Gauss method high-level function for orbit determination of Earth satellites
    from IOD-formatted ra/dec tracking data. IOD angle subformat 2 is assumed.
    Roots of 8-th order Gauss polynomial are computed using np.roots function.
    Note that if `r2_root_ind_vec` is not specified by the user, then the first
    positive root returned by np.roots is used by default.

       Args:
           filename (string): path to IOD-formatted observation data file
           obs_arr (int vector): line numbers in data file to be processed
           bodyname (string): user-defined name of satellite
           refiters (int): number of refinement iterations to be performed
           r2_root_ind_vec (1xlen(obs_arr) int array): indices of Gauss polynomial roots.
           plot (bool): if True, plots data.

       Returns:
           x (tuple): set of Keplerian orbital elements (a, e, taup, omega, I, omega, T)
    """
    # load IOD data for a given satellite
    iod_object_data = load_iod_data(filename)

    # handle default behavior for obs_arr
    if obs_arr is None:
        obs_arr = list(range(1, len(iod_object_data)+1))
    # #the total number of observations used
    nobs = len(obs_arr)

    # get object name
    if bodyname is None:
        bodyname = iod_object_data['object'][obs_arr[0]-1].decode()

    #load data of listed observatories (longitude, latitude, elevation)
    sat_observatories_data = load_sat_observatories_data('sat_tracking_observatories.txt')

    # Earth's G*m value
    mu = mu_Earth

    # if r2_root_ind_vec was not specified, then use always the first positive root by default
    if r2_root_ind_vec is None:
        r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)


    time_vec_list = []
    radius_vec_list = []
    velocity2_vec_list = []
    index_vec_list = []


    counter_process = 0
    end_obs_1 = nobs-2
    for i in range (0, end_obs_1):

        if mode_of_observationsequence == 0:
            end_obs_2 = i + 2
        else:
            end_obs_2 = nobs - 1

        for j in range(i+1, end_obs_2):

            if mode_of_observationsequence == 0:
                end_obs_3 = j + 2
            else:
                end_obs_3 = nobs

            for k in range(j+1, end_obs_3):

                # Apply Gauss method to three elements of data
                inds = [obs_arr[i]-1, obs_arr[j]-1, obs_arr[k]-1]
                print('Processing observation #', counter_process)
                r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_sr, rho_2_sr, rho_3_sr, obs_t , refinement_success = \
                    gauss_iterator_sat(iod_object_data, sat_observatories_data, inds, refiters=refiters, r2_root_ind=r2_root_ind_vec[i])


                # storing all solutions now
                # todo: checking if solutions with radii inside the earth surface can be filtered out?
                # todo: checking if solutuins with v2 velocities higher than escape velocities of earth can be filtered out?
                radius_vec_list.append([r1, r2, r3])
                velocity2_vec_list.append(v2)
                time_vec_list.append(obs_t)
                index_vec_list.append(inds)
                print(inds, np.linalg.norm(v2))

                counter_process += 1


    # serialize results
    time_series = []
    radius_abs_series = []
    radius_vec_series = []

    for measurment in range(len(radius_vec_list)):
        # for r1
        time_series.append(time_vec_list[measurment][0])
        radius_abs_series.append(np.linalg.norm(radius_vec_list[measurment][0]))
        radius_vec_series.append(radius_vec_list[measurment][0])

        # for r2
        time_series.append(time_vec_list[measurment][1])
        radius_abs_series.append(np.linalg.norm(radius_vec_list[measurment][1]))
        radius_vec_series.append(radius_vec_list[measurment][1])

        # for r3
        time_series.append(time_vec_list[measurment][2])
        radius_abs_series.append(np.linalg.norm(radius_vec_list[measurment][2]))
        radius_vec_series.append(radius_vec_list[measurment][2])
        
        

    time_unique = np.unique(time_series)
    radius_mean_vec = []
    radius_mean_abs = []

    for i in range(len(time_unique)):
        timesteps_index = np.where(time_series==time_unique[i])[0]
        radius_x_bytime = [radius_vec_series[i][0] for i in timesteps_index]
        radius_y_bytime = [radius_vec_series[i][1] for i in timesteps_index]
        radius_z_bytime = [radius_vec_series[i][2] for i in timesteps_index]

        radius_x_bytime_mean = np.mean(radius_x_bytime)
        radius_y_bytime_mean = np.mean(radius_y_bytime)
        radius_z_bytime_mean = np.mean(radius_z_bytime)

        radius_mean_vec.append([radius_x_bytime_mean, radius_y_bytime_mean, radius_z_bytime_mean])
        radius_mean_abs.append(np.linalg.norm([radius_x_bytime_mean, radius_y_bytime_mean, radius_z_bytime_mean]))



    # finding the center of points per unique time step, based on the radius length.
    # the polyfitter radius length per time step will be used to scale the directions of the radius vectors.
    # this way the mean direction is kept by all measurements, and the altitude is edjusted by the new radius of the
    # polyfit

    order = 2 # higher than a line (order=1), better number should be discussed
    print(time_series, radius_abs_series)
    p = np.polyfit(time_series, radius_abs_series, order)
    f = np.poly1d(p)

    radius_poly_abs = f(time_unique)
    radius_poly_vec = []
    for i in range(len(radius_mean_vec)):
        radius_poly_vec.append(np.multiply(np.divide(radius_mean_vec[i], radius_mean_abs[i]), radius_poly_abs[i]))



    # PLOT
    if plot:

        v = []
        t = []
        for i in range(len(velocity2_vec_list)):
            v.append(np.linalg.norm(velocity2_vec_list[i]))
            t.append(time_vec_list[i][0])
        plt.plot(t, v, "o", label="all measurements")
        plt.title("Velocity over Time: "+bodyname)
        plt.xlabel("Time [JD]")
        plt.ylabel("Velocity [km/s]")
        plt.legend()
        plt.grid()
        plt.show()


        plt.plot(time_series, np.add(radius_abs_series, -Re), "o", label="all measurements")
        plt.plot(time_unique, np.add(radius_mean_abs, -Re), "o-", label="mean")
        plt.plot(time_unique, np.add(radius_poly_abs, -Re), "*-", label="polyfitted")
        plt.title("Altitude over Time: "+bodyname)
        plt.xlabel("Time [JD]")
        plt.ylabel("Altitude (above SL) [km]")
        plt.legend()
        plt.grid()
        plt.show()


        x_vec = []
        y_vec = []
        z_vec = []

        for i in range(len(radius_poly_vec)):
            x_vec.append(radius_poly_vec[i][0])
            y_vec.append(radius_poly_vec[i][1])
            z_vec.append(radius_poly_vec[i][2])

        ax = plt.axes(aspect='equal', projection='3d')

        # Earth-centered orbits: satellite orbit and geocenter
        ax.scatter3D(0.0, 0.0, 0.0, color='blue', label='Earth')
        ax.scatter3D(x_vec, y_vec, z_vec, color='red', marker='+', label=bodyname+' orbit')
        plt.legend()
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
        ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) #, ncol=3)
        ax.set_title('Angles-only orbit determ. (Gauss): '+bodyname)
        plt.show()

    return time_vec_list, radius_vec_list, velocity2_vec_list, index_vec_list, time_unique, radius_poly_vec

# TODO: evaluate Earth ephemeris only once for a given TDB instant
#       this implies saving all UTC times and their TDB equivalencies
# TODO: allow user to specify ephemerides; currently de432s is always used

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, help="path to IOD-formatted data file", default='../example_data/SATOBS-ML-19200716.txt')
    parser.add_argument('-o', '--obs_array', help="list of lines in file to be read", type=str, default=None)
    parser.add_argument('-b', '--body_name', type=str, help="observed object/body name", default=None)
    parser.add_argument('-r', '--root_index', nargs='*', help="user selection for multiple roots of Gauss polynomial (see docs for more information)", default=None)
    parser.add_argument('-i', '--iterations', type=int, help="number of iterations of Gauss method refinement", default=0)
    parser.add_argument('-p', '--plot', default=True, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()

if __name__ == "__main__":

    args = read_args()
    if args.obs_array is None:
        gauss_method_sat(args.file_path, bodyname=args.body_name,
                         r2_root_ind_vec=args.root_index, refiters=args.iterations, plot=args.plot)
    else:
        obs_arr = [int(item) for item in args.obs_array.split(',')]
        gauss_method_sat(args.file_path, obs_arr=obs_arr, bodyname=args.body_name,
                         r2_root_ind_vec=args.root_index, refiters=args.iterations, plot=args.plot)

