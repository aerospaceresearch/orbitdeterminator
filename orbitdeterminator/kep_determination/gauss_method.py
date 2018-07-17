"""Implements Gauss' method for orbit determination from three topocentric
    angular measurements of celestial bodies.
"""

import math
import numpy as np
from jplephem.spk import SPK
import matplotlib.pyplot as plt
from least_squares import xyz_frame_
import astropy.coordinates
from astropy import units as uts
from astropy.time import Time
from datetime import datetime
# from poliastro.stumpff import c2, c3

def load_mpc_observatories_data(mpc_observatories_fname):
    obs_dt = 'S3, f8, f8, f8, S48'
    obs_delims = [4,10,9,10,48]
    return np.genfromtxt(mpc_observatories_fname, dtype=obs_dt, names=True, delimiter=obs_delims, autostrip=True, encoding=None)

def get_observatory_data(observatory_code, mpc_observatories_data):
    # print('observatory_code = ', observatory_code)
    # print('mpc_observatories_data[\'Code\'] = ', mpc_observatories_data['Code'])
    arr_index = np.where(mpc_observatories_data['Code'] == observatory_code)
    # print('arr_index = ', arr_index)
    # print('mpc_observatories_data[arr_index] = ', mpc_observatories_data[arr_index])
    return arr_index, mpc_observatories_data[arr_index]

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
    return np.mod(6.656306 + 0.0657098242*(jd0-2445700.5) + 1.0027379093*(ut*24), 24.0)

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
def observerpos_mpc(long, parallax_s, parallax_c, jd0, ut):

    # jd0_ = 2444970.5
    # ut_ = 0.89245897
    # # compute Greenwich mean sidereal time (in hours) at UT instant of JD0 date:
    gmst_hrs = gmst(jd0, ut)
    # # compute local sidereal time from GMST and longitude EAST of Greenwich:
    lmst_hrs = localsidtime(gmst_hrs, long)
    # Earth's equatorial radius in kilometers
    Re = 6378.0 # km

    # compute geocentric, Earth-fixed, position of observing site
    # long_site_rad = np.deg2rad(long)
    # x_site = Re*parallax_c*np.cos(long_site_rad)
    # y_site = Re*parallax_c*np.sin(long_site_rad)
    # z_site = Re*parallax_s

    # construct EarthLocation object associated to observing site
    # el_site = astropy.coordinates.EarthLocation.from_geocentric(x_site*uts.km, y_site*uts.km, z_site*uts.km)
    # print('el_site = ', el_site)
    # construct Time object associated to Julian date jd0+ut at observing site
    # print('jd0 = ', jd0)
    # print('ut = ', ut)
    # t_site = Time(jd0+ut, format='jd', location=el_site)
    t_site = Time(jd0+ut, format='jd')
    # print('t_site = ', t_site)
    # print('t_site.location.lon = ', t_site.location.lon )
    # print('t_site.to_datetime = ', t_site.to_datetime()  )
    # get local mean sidereal time
    # t_site_lmst = t_site.sidereal_time('mean')
    long_site = astropy.coordinates.Longitude(long, uts.degree, wrap_angle=360.0*uts.degree)
    # print('str(long)+\'d\' = ', str(long)+'d')
    # print('long_site = ', long_site)
    t_site_lmst = t_site.sidereal_time('mean', longitude=long_site)
    # print('t_site_lmst = ', t_site_lmst)
    lmst_hrs2 = t_site_lmst.value # hours
    # print('gmst_hrs = ', gmst_hrs)
    # print('gmst_hrs2 = ', t_site.sidereal_time('mean', 'greenwich').value)
    # print('lmst_hrs = ', lmst_hrs)
    # print('lmst_hrs2 = ', lmst_hrs2)
    lmst_rad = np.deg2rad(lmst_hrs*15.0) # radians

    # compute cartesian components of geocentric (non rotating) observer position
    x_gc = Re*parallax_c*np.cos(lmst_rad)
    y_gc = Re*parallax_c*np.sin(lmst_rad)
    z_gc = Re*parallax_s

    # print('x_gc = ', x_gc)
    # print('y_gc = ', y_gc)
    # print('z_gc = ', z_gc)

    return np.array((x_gc,y_gc,z_gc))

#observerpos_sat: compute the geocentric position of observer (Earth-centered orbit)
#phi_deg: geodetic latitude (phi), degrees
#altitude_km: altitude above reference ellipsoid, kilometers
#f: Earth's flattening/oblateness factor (adimensional)
#lst_deg: local sidereal time, degrees
def observerpos_sat(phi_deg, altitude_km, f, lst_deg):
    # Earth's equatorial radius in kilometers
    Re = 6378.0 #Earth's radius, km
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

#ra must be in hrs, dec must be in deg
def cosinedirectors(ra_hrs, dec_deg):
    ra_rad = np.deg2rad(ra_hrs*15.0)
    dec_rad = np.deg2rad(dec_deg)

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

# alpha = 1/a
def alpha(x, y, z, u, v, w, mu):
    myRadius=np.sqrt((x**2)+(y**2)+(z**2))
    myVelSqr=(u**2)+(v**2)+(w**2)
    return (2.0/myRadius)-(myVelSqr/mu)

def stumpffS(z):
    if z>0:
        sqrtz = np.sqrt(z)
        return (sqrtz-np.sin(sqrtz))/(sqrtz**3)
    elif z<0:
        sqrtz = np.sqrt(-z)
        # print('sqrtz = ', sqrtz)
        return (np.sinh(sqrtz)-sqrtz)/(sqrtz**3)
    elif z==0:
        return 1.0/6.0

def stumpffC(z):
    if z>0:
        sqrtz = np.sqrt(z)
        return (1.0-np.cos(sqrtz))/z
    elif z<0:
        sqrtz = np.sqrt(-z)
        # print('z = ', z)
        # print('sqrtz = ', sqrtz)
        return (np.cosh(sqrtz)-1.0)/(-z)
    elif z==0:
        return 1.0/2.0

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
        C_z_i = stumpffC(z_i)
        S_z_i = stumpffS(z_i)
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
    return 1.0-(xi**2)*stumpffC(z)/r

def lagrangeg_(tau, xi, z, mu):
    return tau-(xi**3)*stumpffS(z)/np.sqrt(mu)

def gauss_polynomial(x, a, b, c):
    return (x**8)+a*(x**6)+b*(x**3)+c

# Implementation of Gauss method for MPC optical observations of NEAs
def gauss_estimate_mpc(mpc_observatories_data, inds, mpc_data_fname, r2guess=np.nan):
    # load JPL DE430 ephemeris SPK kernel, including TT-TDB difference
    kernel = SPK.open('de430t.bsp')

    # print(kernel)

    # load MPC data for a given NEA
    x = load_data_mpc(mpc_data_fname)

    # print('x[\'ra_hr\'] = ', x['ra_hr'][0:10])
    # print('x[\'ra_min\'] = ', x['ra_min'][0:10]/60.0)
    # print('x[\'ra_sec\'] = ', x['ra_sec'][0:10]/3600.0)
    # print('ra  (hrs) = ', x['ra_hr'][6:18]+x['ra_min'][6:18]/60.0+x['ra_sec'][6:18]/3600.0)
    # print('dec (deg) = ', x['dec_deg'][6:18]+x['dec_min'][6:18]/60.0+x['dec_sec'][6:18]/3600.0)

    # ind_0 = 1409 #0
    # ind_delta = 10
    # ind_end = ind_0+31 #1409

    # print('INPUT DATA FROM MPC:\n', x[ inds ], '\n')

    ra_hrs = x['ra_hr'][inds]+x['ra_min'][inds]/60.0+x['ra_sec'][inds]/3600.0
    dec_deg = x['dec_deg'][inds]+x['dec_min'][inds]/60.0+x['dec_sec'][inds]/3600.0
    # ra_hrs = np.array((43.537,54.420,64.318))/15.0
    # dec_deg = np.array((-8.7833,-12.074,-15.105))

    # cosacosd
    # sinacosd
    # sind

    # ra_rad = np.deg2rad(ra_hrs*15.0)
    # dec_rad = np.deg2rad(dec_deg)

    # print('ra_rad = ', ra_rad)
    # print('dec_rad = ', dec_rad)

    # cosa_cosd = np.cos(ra_rad)*np.cos(dec_rad)
    # sina_cosd = np.sin(ra_rad)*np.cos(dec_rad)
    # sind = np.sin(dec_rad)

    # print('cosa_cosd = ', cosa_cosd)
    # print('sina_cosd = ', sina_cosd)
    # print('sind = ', sind)

    rho1 = cosinedirectors(ra_hrs[0], dec_deg[0])
    rho2 = cosinedirectors(ra_hrs[1], dec_deg[1])
    rho3 = cosinedirectors(ra_hrs[2], dec_deg[2])

    # print('rho1 = ', rho1)
    # print('rho2 = ', rho2)
    # print('rho3 = ', rho3)

    jd01 = Time( datetime(x['yr'][inds[0]], x['month'][inds[0]], x['day'][inds[0]]) ).jd
    jd02 = Time( datetime(x['yr'][inds[1]], x['month'][inds[1]], x['day'][inds[1]]) ).jd
    jd03 = Time( datetime(x['yr'][inds[2]], x['month'][inds[2]], x['day'][inds[2]]) ).jd

    ut1 = x['utc'][inds[0]]
    ut2 = x['utc'][inds[1]]
    ut3 = x['utc'][inds[2]]

    # print('ut1 = ', ut1)
    # print('ut2 = ', ut2)
    # print('ut3 = ', ut3)

    jd1 = jd01+ut1
    jd2 = jd02+ut2
    jd3 = jd03+ut3

    t_jd1_utc = Time(jd1, format='jd', scale='utc')
    t_jd2_utc = Time(jd2, format='jd', scale='utc')
    t_jd3_utc = Time(jd3, format='jd', scale='utc')

    t_jd1_tdb_val = t_jd1_utc.tdb.value
    t_jd2_tdb_val = t_jd2_utc.tdb.value
    t_jd3_tdb_val = t_jd3_utc.tdb.value

    # print(' t_jd1 (utc) = ', t_jd1_utc)
    # print(' t_jd2 (utc) = ', t_jd2_utc)
    # print(' t_jd3 (utc) = ', t_jd3_utc)

    # print(' t_jd1 (tdb) = ', t_jd1_tdb_val)
    # print(' t_jd2 (tdb) = ', t_jd2_tdb_val)
    # print(' t_jd3 (tdb) = ', t_jd3_tdb_val)

    au = 1.495978707e8

    Ea_hc_pos = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    Ea_jd1 = kernel[3,399].compute(t_jd1_tdb_val) + kernel[0,3].compute(t_jd1_tdb_val) - kernel[0,10].compute(t_jd1_tdb_val)
    Ea_jd2 = kernel[3,399].compute(t_jd2_tdb_val) + kernel[0,3].compute(t_jd2_tdb_val) - kernel[0,10].compute(t_jd2_tdb_val)
    Ea_jd3 = kernel[3,399].compute(t_jd3_tdb_val) + kernel[0,3].compute(t_jd3_tdb_val) - kernel[0,10].compute(t_jd3_tdb_val)

    Ea_hc_pos[0] = Ea_jd1/au
    Ea_hc_pos[1] = Ea_jd2/au
    Ea_hc_pos[2] = Ea_jd3/au

    # print('Ea_hc_pos[0] = ', Ea_hc_pos[0])
    # print('Ea_hc_pos[1] = ', Ea_hc_pos[1])
    # print('Ea_hc_pos[2] = ', Ea_hc_pos[2])

    # print('range_ea = ', np.linalg.norm(Ea_jd1, ord=2)/au)

    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    # R[0] = Ea_jd1 + observerpos_mpc(long_691, C_691, S_691, jd01, ut1)
    # R[1] = Ea_jd2 + observerpos_mpc(long_691, C_691, S_691, jd02, ut2)
    # R[2] = Ea_jd3 + observerpos_mpc(long_691, C_691, S_691, jd03, ut3)

    # print('x[\'observatory\'][inds[0]] = ', x['observatory'][inds[0]])
    # print('mpc_observatories_data = ', mpc_observatories_data)
    data_OBS_1 = get_observatory_data(x['observatory'][inds[0]], mpc_observatories_data)
    # print('data_OBS_1 = ', data_OBS_1[1])
    data_OBS_2 = get_observatory_data(x['observatory'][inds[1]], mpc_observatories_data)
    # print('data_OBS_2 = ', data_OBS_2[1])
    data_OBS_3 = get_observatory_data(x['observatory'][inds[2]], mpc_observatories_data)
    # print('data_OBS_3 = ', data_OBS_3[1])

    R[0] = (  Ea_jd1 + observerpos_mpc(data_OBS_1[1]['Long'][0], data_OBS_1[1]['sin'][0], data_OBS_1[1]['cos'][0], jd01, ut1)  )/au
    R[1] = (  Ea_jd2 + observerpos_mpc(data_OBS_2[1]['Long'][0], data_OBS_2[1]['sin'][0], data_OBS_2[1]['cos'][0], jd02, ut2)  )/au
    R[2] = (  Ea_jd3 + observerpos_mpc(data_OBS_3[1]['Long'][0], data_OBS_3[1]['sin'][0], data_OBS_3[1]['cos'][0], jd03, ut3)  )/au

    # print('R[0] = ', R[0])
    # print('R[1] = ', R[1])
    # print('R[2] = ', R[2])

    # make sure time units are consistent!
    tau1 = ((jd01+ut1)-(jd02+ut2)) #*86400.0
    tau3 = ((jd03+ut3)-(jd02+ut2)) #*86400.0
    tau = (tau3-tau1)
    # tau1 = 0-118.10
    # tau3 = 237.58-118.10
    # tau = (tau3-tau1)

    # print('tau1 = ', tau1)
    # print('tau3 = ', tau3)
    # print('tau = ', tau)

    p = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    p[0] = np.cross(rho2, rho3)
    p[1] = np.cross(rho1, rho3)
    p[2] = np.cross(rho1, rho2)

    #print('p = ', p)

    # print('p[0] = ', p[0])
    # print('p[1] = ', p[1])
    # print('p[2] = ', p[2])

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

    # mu_Sun = 132712440041.939400 # Sun's G*m, km^3/seg^2
    mu_Sun = 0.295912208285591100E-03 # Sun's G*m, au^3/day^2
    mu = mu_Sun
    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    # mu = mu_Earth

    a = -(A**2+2.0*A*E+Rsub2p2)
    b = -2.0*mu*B*(A+E)
    c = -(mu**2)*(B**2)

    # print('a = ', a)
    # print('b = ', b)
    # print('c = ', c)

    # plot Gauss function in order to obtain a first estimate of a feasible root
    # x_vals = np.arange(0.0, 2.0*au, 0.001*au)
    # f_vals = gauss_polynomial(x_vals, a, b, c)
    # plt.plot(x_vals/au, f_vals/1e60)
    # plt.show()

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

    if np.isnan(r2guess):
        r2_star = np.real(gauss_poly_roots[rt_indx[0][len(rt_indx[0])-1]])
    else:
        r2_star = np.real(gauss_poly_roots[rt_indx[0][len(rt_indx[0])-1]])
    print('r2_star = ', r2_star)


    # r2_star = newton(gauss_polynomial, np.real(gauss_poly_roots[rt_indx])[0], args=(a, b, c)) #1.06*au)
    # r2_star = newton(gauss_polynomial, 9000.0, args=(a, b, c)) #1.06*au)
    #r2_star = 1.06*au

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

    return r1, r2, r3, v2, jd02+ut2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos, rho_1_, rho_2_, rho_3_

# Refinement stage of Gauss method for MPC ra-dec observations
# INPUT: tau1, tau3, r2, v2, mu, atol, D, R, rho1, rho2, rho3
# OUTPUT: updated r1, r2, v3, v2
def gauss_refinement_mpc(tau1, tau3, r2, v2, atol, D, R, rho1, rho2, rho3, f_1, g_1, f_3, g_3):
    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    # mu = mu_Earth
    # mu_Sun = 132712440041.939400 # Sun's G*m, km^3/seg^2
    mu_Sun = 0.295912208285591100E-03 # Sun's G*m, au^3/day^2
    mu = mu_Sun
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

# Implementation of Gauss method for ra-dec observations of Earth satellites
def gauss_estimate_sat(phi_deg, altitude_km, f, ra_hrs, dec_deg, lst_deg, t_sec):

    rho1 = cosinedirectors(ra_hrs[0], dec_deg[0])
    rho2 = cosinedirectors(ra_hrs[1], dec_deg[1])
    rho3 = cosinedirectors(ra_hrs[2], dec_deg[2])

    # print('rho1 = ', rho1)
    # print('rho2 = ', rho2)
    # print('rho3 = ', rho3)

    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    R[0] = observerpos_sat(phi_deg, altitude_km, f, lst_deg[0])
    R[1] = observerpos_sat(phi_deg, altitude_km, f, lst_deg[1])
    R[2] = observerpos_sat(phi_deg, altitude_km, f, lst_deg[2])

    # print('R[0] = ', R[0])
    # print('R[1] = ', R[1])
    # print('R[2] = ', R[2])

    # make sure time units are consistent!
    tau1 = t_sec[0]-t_sec[1]
    tau3 = t_sec[2]-t_sec[1]
    tau = (tau3-tau1)

    # print('tau1 = ', tau1)
    # print('tau3 = ', tau3)
    # print('tau = ', tau)

    p = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    p[0] = np.cross(rho2, rho3)
    p[1] = np.cross(rho1, rho3)
    p[2] = np.cross(rho1, rho2)

    #print('p = ', p)

    # print('p[0] = ', p[0])
    # print('p[1] = ', p[1])
    # print('p[2] = ', p[2])

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

    mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    mu = mu_Earth

    a = -(A**2+2.0*A*E+Rsub2p2)
    b = -2.0*mu*B*(A+E)
    c = -(mu**2)*(B**2)

    # print('a = ', a)
    # print('b = ', b)
    # print('c = ', c)

    # plot Gauss function in order to obtain a first estimate of a feasible root
    # x_vals = np.arange(0.0, 15000.0, 500.0)
    # f_vals = gauss_polynomial(x_vals, a, b, c)
    # plt.plot(x_vals, f_vals)
    # plt.show()

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
    r2_star = np.real(gauss_poly_roots[rt_indx[0][len(rt_indx[0])-1]])

    # print('r2_star = ', r2_star)

    num1 = 6.0*(D[2,0]*(tau1/tau3)+D[1,0]*(tau/tau3))*(r2_star**3)+mu*D[2,0]*(tau**2-tau1**2)*(tau1/tau3)
    den1 = 6.0*(r2_star**3)+mu*(tau**2-tau3**2)

    rho_1_ = ((num1/den1)-D[0,0])/D0

    rho_2_ = A+(mu*B)/(r2_star**3)

    num3 = 6.0*(D[0,2]*(tau3/tau1)-D[1,2]*(tau/tau1))*(r2_star**3)+mu*D[0,2]*(tau**2-tau3**2)*(tau3/tau1)
    den3 = 6.0*(r2_star**3)+mu*(tau**2-tau1**2)

    rho_3_ = ((num3/den3)-D[2,2])/D0

    # print('rho_1_ = ', rho_1_)
    # print('rho_2_ = ', rho_2_)
    # print('rho_3_ = ', rho_3_)

    r1 = R[0]+rho_1_*rho1
    r2 = R[1]+rho_2_*rho2
    r3 = R[2]+rho_3_*rho3

    # print('r1 = ', r1)
    # print('r2 = ', r2)
    # print('r3 = ', r3)

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

    return r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_

# Refinement stage of Gauss method for Earth satellites rad-dec observations
# INPUT: tau1, tau3, r2, v2, mu, atol, D, R, rho1, rho2, rho3
# OUTPUT: updated r1, r2, v3, v2
def gauss_refinement_sat(tau1, tau3, r2, v2, atol, D, R, rho1, rho2, rho3, f_1, g_1, f_3, g_3):
    mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    mu = mu_Earth
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

def gauss_method_sat(phi_deg, altitude_km, f, ra_hrs, dec_deg, lst_deg, t_sec, refiters=0):
    r1, r2, r3, v2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, rho_1_, rho_2_, rho_3_ = gauss_estimate_sat(phi_deg, altitude_km, f, ra_hrs, dec_deg, lst_deg, t_sec)
    # Apply refinement to Gauss' method, `refiters` iterations
    for i in range(0, refiters):
        r1, r2, r3, v2, rho_1_, rho_2_, rho_3_, f1, g1, f3, g3 = gauss_refinement_sat(tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)
    return r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_

def gauss_method_mpc(mpc_observatories_data, inds_, mpc_data_fname, refiters=0):
    r1, r2, r3, v2, jd2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos, rho_1_, rho_2_, rho_3_ = gauss_estimate_mpc(mpc_observatories_data, inds_, mpc_data_fname)
    # Apply refinement to Gauss' method, `refiters` iterations
    for i in range(0,refiters):
        # print('i = ', i)
        a_local = semimajoraxis(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], 0.295912208285591100E-03)
        e_local = eccentricity(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], 0.295912208285591100E-03)
        if a_local < 0.0 or e_local > 1.0:
            continue
        r1, r2, r3, v2, rho_1_, rho_2_, rho_3_, f1, g1, f3, g3 = gauss_refinement_mpc(tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)
        # print('*r2 = ', r2)
        # print('*v2 = ', v2)
    return r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_, Ea_hc_pos

##############################
if __name__ == "__main__":

    # # Examples 5.11 and 5.12 from book
    # phi_deg = 40.0 # deg
    # altitude_km = 1.0 # km
    # f = 0.003353
    # ra_deg = np.array((43.537, 54.420, 64.318))
    # ra_hrs = ra_deg/15.0
    # dec_deg = np.array((-8.7833, -12.074, -15.105))
    # lst_deg = np.array((44.506, 45.000, 45.499))
    # t_sec = np.array((0.0, 118.10, 237.58))

    # # print('r2 = ', r2)
    # # print('v2 = ', v2)

    # # for i in range(0,6):
    # #     # print('i = ', i)
        

    # r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_ = gauss_method_sat(phi_deg, altitude_km, f, ra_hrs, dec_deg, lst_deg, t_sec, refiters=0)

    # print('r2 = ', r2)
    # print('v2 = ', v2)

    # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
    # mu = mu_Earth

    # a = semimajoraxis(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
    # e = eccentricity(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
    # I = np.rad2deg( inclination(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
    # W = np.rad2deg( longascnode(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
    # w = np.rad2deg( argperi(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu) )
    # theta = np.rad2deg( trueanomaly(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu) )

    # print('a = ', a)
    # print('e = ', e)
    # print('I = ', I, 'deg')
    # print('W = ', W, 'deg')
    # print('w = ', w, 'deg')
    # print('theta = ', theta, 'deg')

    # npoints = 1000
    # theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
    # x_orb_vec = np.zeros((npoints,))
    # y_orb_vec = np.zeros((npoints,))
    # z_orb_vec = np.zeros((npoints,))

    # for i in range(0,npoints):
    #     recovered_xyz = xyz_frame_(a, e, theta_vec[i], np.deg2rad(w), np.deg2rad(I), np.deg2rad(W))
    #     x_orb_vec[i] = recovered_xyz[0]
    #     y_orb_vec[i] = recovered_xyz[1]
    #     z_orb_vec[i] = recovered_xyz[2]

    # ##############################

    au = 1.495978707e8

    # obs_arr = list(range(0,4))+list(range(7,88))+list(range(93,310))+list(range(335,976))+list(range(985,1102))+list(range(1252,1260))
    # obs_arr = list(range(0,4))+list(range(7,110))
    # obs_arr = [1114, 1136, 1251]
    
    # obs_arr = list(range(986,1249))
    # obs_arr = list(range(986,990))
    # obs_arr = list(range(0,4))+list(range(7,11))
    # obs_arr = list(range(335,340))
    # obs_arr = list(range(7,15))
    obs_arr = list(range(860,978))
    nobs = len(obs_arr)
    print('nobs = ', nobs)
    print('obs_arr = ', obs_arr)

    x_vec = np.zeros((nobs-2,))
    y_vec = np.zeros((nobs-2,))
    z_vec = np.zeros((nobs-2,))

    x_Ea_vec = np.zeros((nobs-2,))
    y_Ea_vec = np.zeros((nobs-2,))
    z_Ea_vec = np.zeros((nobs-2,))

    a_vec = np.zeros((nobs,))
    e_vec = np.zeros((nobs,))
    I_vec = np.zeros((nobs,))
    W_vec = np.zeros((nobs,))
    w_vec = np.zeros((nobs,))

    r2s_guess_vec = np.zeros((nobs-2,))


    mpc_observatories_data = load_mpc_observatories_data('mpc_observatories.txt')

    ###########################
    print('len(range (0,nobs-2)) = ', len(range (0,nobs-2)))
    for j in range (0,nobs-2):
        # Apply Gauss method to three elements of data
        # inds_ = [1409, 1442, 1477] #[10,1,2] # [1409,1440,1477]
        ind0 = obs_arr[j]
        # inds_ = [obs_arr[j], obs_arr[j+1], obs_arr[j+2]]
        inds_ = [ind0, ind0+1, ind0+2]
        print('j = ', j)
        r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_, Ea_hc_pos = gauss_method_mpc(mpc_observatories_data, inds_, '../example_data/mpc_data.txt', refiters=5)

        # print('|r1| = ', np.linalg.norm(r1,ord=2))
        # print('|r2| = ', np.linalg.norm(r2,ord=2))
        # print('|r3| = ', np.linalg.norm(r3,ord=2))
        # print('r2 = ', r2)
        # print('v2 = ', v2)

        # print("*** CARTESIAN STATES AND REFERENCE EPOCH ***")

        # print('r2 = ', r2, 'km')
        # print('v2 = ', v2, 'km/s')

        # print('r2 = ', r2/au, 'au')
        # print('v2 = ', v2*86400/au, 'au/day')
        # print('JD2 = ', jd2, '\n')

        # r2_au = r2/au
        # v2_au_day = v2*86400/au
        # mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
        # mu = mu_Earth
        # mu_Sun = 132712440041.939400 # Sun's G*m, km^3/seg^2
        mu_Sun = 0.295912208285591100E-03 # Sun's G*m, au^3/day^2
        mu = mu_Sun

        a_num = semimajoraxis(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
        e_num = eccentricity(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)

        if 0.0<=e_num<=1.0 and a_num>=0.0:
            a_vec[j] = a_num
            e_vec[j] = e_num
            I_vec[j] = np.rad2deg( inclination(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
            W_vec[j] = np.rad2deg( longascnode(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
            w_vec[j] = np.rad2deg( argperi(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu) )
            # print(a_vec[j], e_vec[j], I_vec[j], W_vec[j], w_vec[j])
            x_vec[j] = r2[0]
            y_vec[j] = r2[1]
            z_vec[j] = r2[2]
            x_Ea_vec[j] = Ea_hc_pos[1][0]
            y_Ea_vec[j] = Ea_hc_pos[1][1]
            z_Ea_vec[j] = Ea_hc_pos[1][2]

        # print(a_num/au, 'au', ', ', e_num)
        # print(a_num, 'au', ', ', e_num)
        # print('j = ', j, 'obs_arr[j] = ', obs_arr[j])

    # print('x_vec = ', x_vec)
    # print('a_vec = ', a_vec)
    # print('e_vec = ', e_vec)
    print('a_vec = ', a_vec)
    print('len(a_vec) = ', len(a_vec))
    print('len(a_vec[a_vec>0.0]) = ', len(a_vec[a_vec>0.0]))

    print('*** AVERAGE ORBITAL ELEMENTS: a, e, I, Omega, omega ***')
    # print('Semimajor axis, a: ', a_, 'km')
    # print(np.mean(a_vec[a_vec>0.0])/au, 'au', ', ', np.mean(e_vec[e_vec<1.0]))
    print(np.mean(a_vec[a_vec>0.0]), 'au', ', ', np.mean(e_vec[a_vec>0.0]), ', ', np.mean(I_vec[a_vec>0.0]), 'deg', ', ', np.mean(W_vec[a_vec>0.0]), 'deg', ', ', np.mean(w_vec[a_vec>0.0]), 'deg')

    ###########################
    # Plot

    from mpl_toolkits import mplot3d
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Apophis plot
    ax.scatter3D(x_vec[x_vec!=0.0], y_vec[x_vec!=0.0], z_vec[x_vec!=0.0], color='red', marker='.', label='Apophis orbit')
    ax.scatter3D(x_Ea_vec[x_Ea_vec!=0.0], y_Ea_vec[x_Ea_vec!=0.0], z_Ea_vec[x_Ea_vec!=0.0], color='blue', marker='.', label='Earth orbit')
    ax.scatter3D(0.0, 0.0, 0.0, color='yellow', label='Sun')
    plt.legend()
    plt.xlabel('x (au)')
    plt.ylabel('y (au)')
    plt.title('Angles-only orbit determ. (Gauss): Apophis')
    plt.show()
    # end, Apophis plot

    # xline1 = np.array((0.0, R[0][0]))
    # yline1 = np.array((0.0, R[0][1]))
    # zline1 = np.array((0.0, R[0][2]))
    # xline2 = np.array((0.0, R[1][0]))
    # yline2 = np.array((0.0, R[1][1]))
    # zline2 = np.array((0.0, R[1][2]))
    # xline3 = np.array((0.0, R[2][0]))
    # yline3 = np.array((0.0, R[2][1]))
    # zline3 = np.array((0.0, R[2][2]))
    # xline4 = np.array((0.0, r1[0]))
    # yline4 = np.array((0.0, r1[1]))
    # zline4 = np.array((0.0, r1[2]))
    # xline5 = np.array((R[0][0], R[0][0]+rho_1_*rho1[0]))
    # yline5 = np.array((R[0][1], R[0][1]+rho_1_*rho1[1]))
    # zline5 = np.array((R[0][2], R[0][2]+rho_1_*rho1[2]))
    # xline6 = np.array((0.0, r2[0]))
    # yline6 = np.array((0.0, r2[1]))
    # zline6 = np.array((0.0, r2[2]))
    # xline7 = np.array((R[1][0], R[1][0]+rho_2_*rho2[0]))
    # yline7 = np.array((R[1][1], R[1][1]+rho_2_*rho2[1]))
    # zline7 = np.array((R[1][2], R[1][2]+rho_2_*rho2[2]))
    # xline8 = np.array((0.0, r3[0]))
    # yline8 = np.array((0.0, r3[1]))
    # zline8 = np.array((0.0, r3[2]))
    # xline9 = np.array((R[2][0], R[2][0]+rho_3_*rho3[0]))
    # yline9 = np.array((R[2][1], R[2][1]+rho_3_*rho3[1]))
    # zline9 = np.array((R[2][2], R[2][2]+rho_3_*rho3[2]))
    # ax.plot3D(xline1, yline1, zline1, 'gray', label='Observer 1')
    # ax.plot3D(xline2, yline2, zline2, 'blue', label='Observer 2')
    # ax.plot3D(xline3, yline3, zline3, 'green', label='Observer 3')
    # ax.plot3D(xline4, yline4, zline4, 'orange')
    # ax.plot3D(xline5, yline5, zline5, 'red', label='LOS 1')
    # ax.plot3D(xline6, yline6, zline6, 'black')
    # ax.plot3D(xline7, yline7, zline7, 'cyan', label='LOS 2')
    # ax.plot3D(xline8, yline8, zline8, 'brown')
    # ax.plot3D(xline9, yline9, zline9, 'yellow', label='LOS 3')
    # ax.scatter3D(0.0, 0.0, 0.0, color='blue', label='Geocenter')
    # ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'black', label='Satellite orbit')
    # # ax.plot_surface(x_ea_surf, y_ea_surf, z_ea_surf, color='b')
    # # ax.set_aspect('equal')
    # plt.legend()
    # ax.set_xlim(-10000.0, 10000.0)
    # ax.set_ylim(-10000.0, 10000.0)
    # ax.set_zlim(-10000.0, 10000.0)
    # ax.set_xlabel('x (km)')
    # ax.set_ylabel('y (km)')
    # ax.set_zlabel('z (km)')
    # plt.title('Satellite orbit determination: Gauss method')
    # plt.show()
