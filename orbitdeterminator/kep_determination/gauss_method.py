"""Implements Gauss' method for orbit determination from three topocentric
    angular measurements of celestial bodies.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from least_squares import xyz_frame_
from astropy.coordinates import Longitude
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import units as uts
from astropy.time import Time
from datetime import datetime, timedelta
# from poliastro.stumpff import c2, c3

np.set_printoptions(precision=16)

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
    return mpc_observatories_data[arr_index[0]]

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
    # mpc_names = ['mpnum','provdesig','discovery','publishnote','j2000','yr','month','day','utc','ra_hr','ra_min','ra_sec','dec_deg','dec_min','dec_sec','9xblank','magband','6xblank','observatory']
    mpc_names = ['mpnum','provdesig','discovery','publishnote','j2000','yr','month','day','utc','radec','9xblank','magband','6xblank','observatory']
    # mpc_delims are the fixed-width column delimiter following MPC format description
    # mpc_delims = [5,7,1,1,1,4,3,3,7,2,3,7,3,3,6,9,6,6,3]
    mpc_delims = [5,7,1,1,1,4,3,3,7,24,9,6,6,3]
    return np.genfromtxt(fname, dtype=dt, names=mpc_names, delimiter=mpc_delims, autostrip=True)

# Compute geocentric observer position at the Julian date jd_utc (UTC)
# at a given observation site defined by its longitude, and parallax constants S and C
# formula taken from top of page 266, chapter 5, Orbital Mechanics book
# the parallax constants S and C are defined by
# S=rho cos phi' C=rho sin phi'
# rho: slant range
# phi': geocentric latitude
# We have the following:
# phi' = atan(S/C)
# rho = sqrt(S**2+C**2)
def observerpos_mpc(long, parallax_s, parallax_c, jd_utc):

    # Earth's equatorial radius in kilometers
    Re = 6378.0 # km

    # define Longitude object for the observation site longitude
    long_site = Longitude(long, uts.degree, wrap_angle=360.0*uts.degree)
    # print('long_site = ', long_site)
    # compute sidereal time of observation at site
    t_site_lmst = jd_utc.sidereal_time('mean', longitude=long_site)
    # print('t_site_lmst = ', t_site_lmst)
    lmst_hrs = t_site_lmst.value # hours
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

# Implementation of Gauss method for MPC optical observations of NEAs
def gauss_estimate_mpc(spk_kernel, mpc_object_data, mpc_observatories_data, inds, r2_root_ind=0):
    # construct SkyCoord 3-element array with observational information
    timeobs = np.zeros((3,), dtype=Time)
    sc = np.zeros((3,), dtype=SkyCoord)

    timeobs[0] = Time( datetime(mpc_object_data['yr'][inds[0]], mpc_object_data['month'][inds[0]], mpc_object_data['day'][inds[0]]) + timedelta(days=mpc_object_data['utc'][inds[0]]) )
    timeobs[1] = Time( datetime(mpc_object_data['yr'][inds[1]], mpc_object_data['month'][inds[1]], mpc_object_data['day'][inds[1]]) + timedelta(days=mpc_object_data['utc'][inds[1]]) )
    timeobs[2] = Time( datetime(mpc_object_data['yr'][inds[2]], mpc_object_data['month'][inds[2]], mpc_object_data['day'][inds[2]]) + timedelta(days=mpc_object_data['utc'][inds[2]]) )

    sc[0] = SkyCoord(mpc_object_data['radec'][inds[0]], unit=(uts.hourangle, uts.deg), obstime=timeobs[0])
    sc[1] = SkyCoord(mpc_object_data['radec'][inds[1]], unit=(uts.hourangle, uts.deg), obstime=timeobs[1])
    sc[2] = SkyCoord(mpc_object_data['radec'][inds[2]], unit=(uts.hourangle, uts.deg), obstime=timeobs[2])

    # print('sc[0] = ', sc[0])
    # print('sc[1] = ', sc[1])
    # print('sc[2] = ', sc[2])

    jd1 = sc[0].obstime.jd
    jd2 = sc[1].obstime.jd
    jd3 = sc[2].obstime.jd

    # print('jd1 (utc) = ', jd1)
    # print('jd2 (utc) = ', jd2)
    # print('jd3 (utc) = ', jd3)

    ### COMPUTE OBSERVER POSITION VECTORS

    # load MPC observatory data
    obsite1 = get_observatory_data(mpc_object_data['observatory'][inds[0]], mpc_observatories_data)
    obsite2 = get_observatory_data(mpc_object_data['observatory'][inds[1]], mpc_observatories_data)
    obsite3 = get_observatory_data(mpc_object_data['observatory'][inds[2]], mpc_observatories_data)
    # print('obsite1 = ', obsite1)
    # print('obsite2 = ', obsite2)
    # print('obsite3 = ', obsite3)

    # astronomical unit in km
    au = 1.495978707e8

    # compute TDB instant of each observation
    t_jd1_tdb_val = sc[0].obstime.tdb.jd
    t_jd2_tdb_val = sc[1].obstime.tdb.jd
    t_jd3_tdb_val = sc[2].obstime.tdb.jd

    # print(' jd1 (tdb) = ', t_jd1_tdb_val)
    # print(' jd2 (tdb) = ', t_jd2_tdb_val)
    # print(' jd3 (tdb) = ', t_jd3_tdb_val)

    Ea_hc_pos = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    Ea_jd1 = spk_kernel[3,399].compute(t_jd1_tdb_val) + spk_kernel[0,3].compute(t_jd1_tdb_val) - spk_kernel[0,10].compute(t_jd1_tdb_val)
    Ea_jd2 = spk_kernel[3,399].compute(t_jd2_tdb_val) + spk_kernel[0,3].compute(t_jd2_tdb_val) - spk_kernel[0,10].compute(t_jd2_tdb_val)
    Ea_jd3 = spk_kernel[3,399].compute(t_jd3_tdb_val) + spk_kernel[0,3].compute(t_jd3_tdb_val) - spk_kernel[0,10].compute(t_jd3_tdb_val)

    Ea_hc_pos[0] = Ea_jd1/au
    Ea_hc_pos[1] = Ea_jd2/au
    Ea_hc_pos[2] = Ea_jd3/au

    # print('Ea_hc_pos[0] = ', Ea_hc_pos[0])
    # print('Ea_hc_pos[1] = ', Ea_hc_pos[1])
    # print('Ea_hc_pos[2] = ', Ea_hc_pos[2])

    # print('range_ea = ', np.linalg.norm(Ea_jd1, ord=2)/au)

    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    R[0] = (  Ea_jd1 + observerpos_mpc(obsite1['Long'][0], obsite1['sin'][0], obsite1['cos'][0], sc[0].obstime)  )/au
    R[1] = (  Ea_jd2 + observerpos_mpc(obsite2['Long'][0], obsite2['sin'][0], obsite2['cos'][0], sc[1].obstime)  )/au
    R[2] = (  Ea_jd3 + observerpos_mpc(obsite3['Long'][0], obsite3['sin'][0], obsite3['cos'][0], sc[2].obstime)  )/au

    # print('R[0] = ', R[0])
    # print('R[1] = ', R[1])
    # print('R[2] = ', R[2])

    ### PERFORM GAUSS METHOD ALGORITHM

    #compute Line-Of-Sight (LOS) vectors
    rho1 = losvector(sc[0].ra.rad, sc[0].dec.rad)
    rho2 = losvector(sc[1].ra.rad, sc[1].dec.rad)
    rho3 = losvector(sc[2].ra.rad, sc[2].dec.rad)

    # print('rho1 = ', rho1)
    # print('rho2 = ', rho2)
    # print('rho3 = ', rho3)

    # compute time differences; make sure time units are consistent!
    tau1 = (jd1-jd2)
    tau3 = (jd3-jd2)
    tau = (tau3-tau1)
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

    return r1, r2, r3, v2, jd2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos, rho_1_, rho_2_, rho_3_

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

    rho1 = losvector(ra_hrs[0], dec_deg[0])
    rho2 = losvector(ra_hrs[1], dec_deg[1])
    rho3 = losvector(ra_hrs[2], dec_deg[2])

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

def gauss_method_mpc(spk_kernel, mpc_object_data, mpc_observatories_data, inds_, refiters=0, r2_root_ind=0):
    r1, r2, r3, v2, jd2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos, rho_1_, rho_2_, rho_3_ = gauss_estimate_mpc(spk_kernel, mpc_object_data, mpc_observatories_data, inds_, r2_root_ind=r2_root_ind)
    # Apply refinement to Gauss' method, `refiters` iterations
    for i in range(0,refiters):
        # print('i = ', i)
        # a_local = semimajoraxis(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], 0.295912208285591100E-03)
        # e_local = eccentricity(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], 0.295912208285591100E-03)
        # if a_local < 0.0 or e_local > 1.0:
        #     continue
        r1, r2, r3, v2, rho_1_, rho_2_, rho_3_, f1, g1, f3, g3 = gauss_refinement_mpc(tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)
        # print('*r2 = ', r2)
        # print('*v2 = ', v2)
    return r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_, Ea_hc_pos

##############################
#if __name__ == "__main__":

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
