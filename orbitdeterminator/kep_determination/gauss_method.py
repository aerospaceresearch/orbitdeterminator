"""Implements Gauss' method for orbit determination from three topocentric
    angular measurements of celestial bodies.
"""

import math
import numpy as np
from jplephem.spk import SPK
import matplotlib.pyplot as plt
from scipy.optimize import newton
# from scipy.optimize import least_squares

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
    x_gc = Re*rho_gc*np.cos(phi_gc)*np.cos(np.deg2rad(15.0*lst_hrs))
    y_gc = Re*rho_gc*np.cos(phi_gc)*np.sin(np.deg2rad(15.0*lst_hrs))
    z_gc = Re*rho_gc*np.sin(phi_gc)

    return np.array((x_gc,y_gc,z_gc))

#ra must be in hrs, dec must be in deg
def cosinedirectors(ra_hrs, dec_deg):
    ra_rad = np.deg2rad(ra_hrs*15.0)
    dec_rad = np.deg2rad(dec_deg)

    cosa_cosd = np.cos(ra_rad)*np.cos(dec_rad)
    sina_cosd = np.sin(ra_rad)*np.cos(dec_rad)
    sind = np.sin(dec_rad)
    return np.array((cosa_cosd, sina_cosd, sind))

# the following function was copied from
# https://gist.github.com/jiffyclub/1294443
def date_to_jd(year,month,day):
    """
    Convert a date to Julian Day.
    
    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
        4th ed., Duffet-Smith and Zwart, 2011.
    
    Parameters
    ----------
    year : int
        Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.
        
    month : int
        Month as integer, Jan = 1, Feb. = 2, etc.
    
    day : float
        Day, may contain fractional part.
    
    Returns
    -------
    jd : float
        Julian Day
        
    Examples
    --------
    Convert 6 a.m., February 17, 1985 to Julian Day
    
    >>> date_to_jd(1985,2,17.25)
    2446113.75
    
    """
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    
    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = math.trunc(yearp / 100.)
        B = 2 - A + math.trunc(A / 4.)
        
    if yearp < 0:
        C = math.trunc((365.25 * yearp) - 0.75)
    else:
        C = math.trunc(365.25 * yearp)
        
    D = math.trunc(30.6001 * (monthp + 1))
    
    jd = B + C + D + day + 1720994.5
    
    return jd

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

#########################

# an example of computation of sidereal times, which will be added as unit testing:
# Determine the true and mean sidereal time on 01 January 1982 at 1h CET for
# Munich (lambda = -11deg36.5', delta_lambda = -15''.476, epsilon = 23deg26'27'')

# jd0_ = 2444970.5 #Julian day of 1982, January 1st
# ut = 0.0
# munich_long = (11.0+36.5/60.0) #degrees
# my_d_lamb = -(15.476)/15.0/3600.0 # hours
# epsilon_ = np.deg2rad( 23.0+26.0/60.0+27.0/3600.0 ) #radians

# print('JD = ', 2444970.5)
# print('dl*cos(e) = ', my_d_lamb*np.cos(epsilon_), 's')

# mu_gmst = gmst(jd0_,ut)
# mu_gmst_hrs = np.floor(mu_gmst)
# mu_gmst_min = np.floor((mu_gmst-mu_gmst_hrs)*60.0)
# mu_gmst_sec = ((mu_gmst-mu_gmst_hrs)*60.0-mu_gmst_min)*60.0

# mu_gast = gast(jd0_, ut, my_d_lamb, epsilon_)
# mu_gast_hrs = np.floor(mu_gast)
# mu_gast_min = np.floor((mu_gast-mu_gast_hrs)*60.0)
# mu_gast_sec = ((mu_gast-mu_gast_hrs)*60.0-mu_gast_min)*60.0

# mu_lmst = localsidtime(mu_gmst, munich_long)
# mu_lmst_hrs = np.floor(mu_lmst)
# mu_lmst_min = np.floor((mu_lmst-mu_lmst_hrs)*60.0)
# mu_lmst_sec = ((mu_lmst-mu_lmst_hrs)*60.0-mu_lmst_min)*60.0

# mu_last = localsidtime(mu_gast, munich_long)
# mu_last_hrs = np.floor(mu_last)
# mu_last_min = np.floor((mu_last-mu_last_hrs)*60.0)
# mu_last_sec = ((mu_last-mu_last_hrs)*60.0-mu_last_min)*60.0

# print('GMST = ', mu_gmst )
# print('mu_gmst_hrs = ', mu_gmst_hrs)
# print('mu_gmst_min = ', mu_gmst_min)
# print('mu_gmst_sec = ', mu_gmst_sec)

# print('GAST = ', mu_gast )
# print('mu_gast_hrs = ', mu_gast_hrs)
# print('mu_gast_min = ', mu_gast_min)
# print('mu_gast_sec = ', mu_gast_sec)

# print('LMST = ', mu_lmst )
# print('mu_lmst_hrs = ', mu_lmst_hrs)
# print('mu_lmst_min = ', mu_lmst_min)
# print('mu_lmst_sec = ', mu_lmst_sec)

# print('LAST = ', mu_last )
# print('mu_last_hrs = ', mu_last_hrs)
# print('mu_last_min = ', mu_last_min)
# print('mu_last_sec = ', mu_last_sec)

#########################
# TODO: implement Gauss' method, from algorithm 5.5, chapter 5, page 285, Orbital Mechanics book
# input: three pairs of topocentric (ra_i, dec_i), three geocentric observer vectors R_i,
#        and three observations times t_i, i= 1,2,3
# output: cartesian state [x0,y0,z0,u0,v0,w0] at reference epoch t0

def gauss_polynomial(x, a, b, c):
    return (x**8)+a*(x**6)+b*(x**3)+c

def gauss_method_hc(mpc_observatories_data, inds, mpc_data_fname):
    # load JPL DE430 ephemeris SPK kernel, including TT-TDB difference
    kernel = SPK.open('de430t.bsp')

    # print(kernel)

    # Julian date of Apophis discovery observations:
    # jd = 2453079.5 # 2004 Mar 15
    # ut = 24.0*0.10789 # UT time of 1st observation

    #geocentric observer position at time of 1st Apophis observation:
    # pos_691 = observerpos(long_691, S_691, C_691, jd, ut)
    # print('pos_691 = ', pos_691)

    # cross-check:
    # radius_ = np.sqrt(pos_691[0]**2+pos_691[1]**2+pos_691[2]**2)
    # print('radius_ = ', radius_)

    # load MPC data for Apophis
    x = load_data_mpc(mpc_data_fname)

    # print('x[\'ra_hr\'] = ', x['ra_hr'][0:10])
    # print('x[\'ra_min\'] = ', x['ra_min'][0:10]/60.0)
    # print('x[\'ra_sec\'] = ', x['ra_sec'][0:10]/3600.0)
    # print('ra  (hrs) = ', x['ra_hr'][6:18]+x['ra_min'][6:18]/60.0+x['ra_sec'][6:18]/3600.0)
    # print('dec (deg) = ', x['dec_deg'][6:18]+x['dec_min'][6:18]/60.0+x['dec_sec'][6:18]/3600.0)

    # ind_0 = 1409 #0
    # ind_delta = 10
    # ind_end = ind_0+31 #1409

    print('INPUT DATA FROM MPC:\n', x[ inds ], '\n')

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

    jd01 = date_to_jd(x['yr'][inds[0]], x['month'][inds[0]], x['day'][inds[0]])
    jd02 = date_to_jd(x['yr'][inds[1]], x['month'][inds[1]], x['day'][inds[1]])
    jd03 = date_to_jd(x['yr'][inds[2]], x['month'][inds[2]], x['day'][inds[2]])

    ut1 = x['utc'][inds[0]]
    ut2 = x['utc'][inds[1]]
    ut3 = x['utc'][inds[2]]

    print(' jd1 = ', jd01+ut1)
    print(' jd2 = ', jd02+ut2)
    print(' jd3 = ', jd03+ut3)

    # print(' ut1 = ', ut1)
    # print(' ut2 = ', ut2)
    # print(' ut3 = ', ut3)

    au = 1.495978707e8

    Ea_hc_pos = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    Ea_jd1 = kernel[3,399].compute(jd01+ut1) + kernel[0,3].compute(jd01+ut1) - kernel[0,10].compute(jd01+ut1)
    Ea_jd2 = kernel[3,399].compute(jd02+ut2) + kernel[0,3].compute(jd02+ut2) - kernel[0,10].compute(jd02+ut2)
    Ea_jd3 = kernel[3,399].compute(jd03+ut3) + kernel[0,3].compute(jd03+ut3) - kernel[0,10].compute(jd03+ut3)

    Ea_hc_pos[0] = Ea_jd1/au
    Ea_hc_pos[1] = Ea_jd2/au
    Ea_hc_pos[2] = Ea_jd3/au

    # print('Ea_jd1 = ', Ea_jd1)
    # print('Ea_jd2 = ', Ea_jd2)
    # print('Ea_jd3 = ', Ea_jd3)

    # print('range_ea = ', np.linalg.norm(Ea_jd1, ord=2)/au)

    R = np.array((np.zeros((3,)),np.zeros((3,)),np.zeros((3,))))

    # R[0] = Ea_jd1 + observerpos(long_691, C_691, S_691, jd01, ut1)
    # R[1] = Ea_jd2 + observerpos(long_691, C_691, S_691, jd02, ut2)
    # R[2] = Ea_jd3 + observerpos(long_691, C_691, S_691, jd03, ut3)

    # print('x[\'observatory\'][inds[0]] = ', x['observatory'][inds[0]])
    # print('mpc_observatories_data = ', mpc_observatories_data)
    data_OBS_1 = get_observatory_data(x['observatory'][inds[0]], mpc_observatories_data)
    print('data_OBS_1 = ', data_OBS_1[1])
    data_OBS_2 = get_observatory_data(x['observatory'][inds[1]], mpc_observatories_data)
    print('data_OBS_2 = ', data_OBS_2[1])
    data_OBS_3 = get_observatory_data(x['observatory'][inds[2]], mpc_observatories_data)
    print('data_OBS_3 = ', data_OBS_3[1])

    R[0] = (  Ea_jd1 + observerpos(data_OBS_1[1]['Long'][0], data_OBS_1[1]['cos'][0], data_OBS_1[1]['sin'][0], jd01, ut1)  )/au
    R[1] = (  Ea_jd2 + observerpos(data_OBS_2[1]['Long'][0], data_OBS_2[1]['cos'][0], data_OBS_2[1]['sin'][0], jd02, ut2)  )/au
    R[2] = (  Ea_jd3 + observerpos(data_OBS_3[1]['Long'][0], data_OBS_3[1]['cos'][0], data_OBS_3[1]['sin'][0], jd03, ut3)  )/au

    # R[0] = np.array((3489.8, 3430.2, 4078.5))
    # R[1] = np.array((3460.1, 3460.1, 4078.5))
    # R[2] = np.array((3429.9, 3490.1, 4078.5))

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

    print('tau1 = ', tau1)
    print('tau3 = ', tau3)
    print('tau = ', tau)

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

    # print('f(0) = ', f_vals[0])

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
    print('rt_indx[0] = ', rt_indx[0])
    # print('np.real(gauss_poly_roots[rt_indx])[0] = ', np.real(gauss_poly_roots[rt_indx])[0])
    if len(rt_indx[0]) > 1: #-1:#
        print('WARNING: Gauss polynomial has more than 1 real, positive solution')
        print('gauss_poly_coeffs = ', gauss_poly_coeffs)
        print('gauss_poly_roots = ', gauss_poly_roots)
        print('len(rt_indx[0]) = ', len(rt_indx[0]))
        print('np.real(gauss_poly_roots[rt_indx[0]]) = ', np.real(gauss_poly_roots[rt_indx[0]]))
    r2_star = np.real(gauss_poly_roots[rt_indx[0][0]])

    # r2_star = newton(gauss_polynomial, np.real(gauss_poly_roots[rt_indx])[0], args=(a, b, c)) #1.06*au)
    # r2_star = newton(gauss_polynomial, 9000.0, args=(a, b, c)) #1.06*au)
    #r2_star = 1.06*au

    # print('r2_star = ', r2_star/au)
    print('r2_star = ', r2_star)

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

    return r1, r2, r3, v2, jd02+ut2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos

# refinement
# INPUT: tau1, tau3, r2, v2, mu, atol, D, R, rho1, rho2, rho3
# OUTPUT: updated r1, r2, v3, v2
def gauss_refinement_hc(tau1, tau3, r2, v2, atol, D, R, rho1, rho2, rho3, f_1, g_1, f_3, g_3):
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

##############################

# longitude and parallax constants C,S for observatory with code 691:
# 248.4010  0.84951  +0.52642
# taken from https://www.minorplanetcenter.net/iau/lists/ObsCodesF.html
# retrieved on: 19 Jun 2018

# long_691 = 248.4010 # degrees
# C_691 = 0.84951
# S_691 = +0.52642

# # 586   0.1423  0.73358  +0.67799  Pic du Midi
# long_586 = 0.1423 # degrees
# C_586 = 0.73358
# S_586 = +0.67799

au = 1.495978707e8

# obs_arr = list(range(0,4))+list(range(7,88))+list(range(93,310))+list(range(335,976))+list(range(985,1102))+list(range(1252,1260))
# obs_arr = list(range(0,4))+list(range(7,110))
obs_arr = list(range(860,978))
nobs = len(obs_arr) #2520-2417 #50 #2 #19
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

mpc_observatories_data = load_mpc_observatories_data('mpc_observatories.txt')

###########################
for j in range (0,nobs-2):
    # Apply Gauss method to three elements of data
    # inds_ = [1409, 1442, 1477] #[10,1,2] # [1409,1440,1477]
    ind0 = obs_arr[j] #0 #2417 #2538 #2475 #55 #7 #0 #1409
    #inds_ = [obs_arr[j], obs_arr[j+1], obs_arr[j+2]]
    inds_ = [obs_arr[j], obs_arr[j]+1, obs_arr[j]+2]
    r1, r2, r3, v2, jd2, D, R, rho1, rho2, rho3, tau1, tau3, f1, g1, f3, g3, Ea_hc_pos = gauss_method_hc(mpc_observatories_data, inds_, '../example_data/mpc_data.txt')
    # Apply refinement to Gauss' method, 100 iterations
    for i in range(0,10):
        # print('i = ', i)
        a_local = semimajoraxis(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], 0.295912208285591100E-03)
        e_local = eccentricity(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], 0.295912208285591100E-03)
        if a_local < 0.0 or e_local > 1.0:
            continue
        r1_, r2, r3_, v2, rho_1_, rho_2_, rho_3_, f1, g1, f3, g3 = gauss_refinement_hc(tau1, tau3, r2, v2, 3e-14, D, R, rho1, rho2, rho3, f1, g1, f3, g3)
        # print(f1, g1, f3, g3)

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
        x_vec[j] = r2[0]
        y_vec[j] = r2[1]
        z_vec[j] = r2[2]
        x_Ea_vec[j] = Ea_hc_pos[1][0]
        y_Ea_vec[j] = Ea_hc_pos[1][1]
        z_Ea_vec[j] = Ea_hc_pos[1][2]

    # print(a_num/au, 'au', ', ', e_num)
    print(a_num, 'au', ', ', e_num)
    print('j = ', j, 'obs_arr[j] = ', obs_arr[j])

print('x_vec = ', x_vec)

print('*** ORBITAL ELEMENTS: a (au), e (adim) ***')
# print('Semimajor axis, a: ', a_, 'km')
# print(np.mean(a_vec[a_vec>0.0])/au, 'au', ', ', np.mean(e_vec[e_vec<1.0]))
print(np.mean(a_vec[a_vec>0.0]), 'au', ', ', np.mean(e_vec[e_vec<1.0]))

###########################
# Plot

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

# ax.scatter3D(x_vec[x_vec!=0.0]/au, y_vec[x_vec!=0.0]/au, z_vec[x_vec!=0.0]/au, color='red', label='Apophis orbit')
ax.scatter3D(x_vec[x_vec!=0.0], y_vec[x_vec!=0.0], z_vec[x_vec!=0.0], color='red', marker='.', label='Apophis orbit')
ax.scatter3D(x_Ea_vec[x_Ea_vec!=0.0], y_Ea_vec[x_Ea_vec!=0.0], z_Ea_vec[x_Ea_vec!=0.0], color='blue', marker=',', label='Earth orbit')
ax.scatter3D(0.0, 0.0, 0.0, color='yellow', label='Sun')
plt.legend()
plt.xlabel('x (au)')
plt.ylabel('y (au)')
plt.title('Heliocentric orbit determination by Gauss method: Apophis')
plt.show()

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
# ax.plot3D(xline1/au, yline1/au, zline1/au, 'gray', label='Observer 1')
# ax.plot3D(xline2/au, yline2/au, zline2/au, 'blue', label='Observer 2')
# ax.plot3D(xline3/au, yline3/au, zline3/au, 'green', label='Observer 3')
# ax.plot3D(xline4/au, yline4/au, zline4/au, 'orange')
# ax.plot3D(xline5/au, yline5/au, zline5/au, 'red', label='LOS 1')
# ax.plot3D(xline6/au, yline6/au, zline6/au, 'black')
# ax.plot3D(xline7/au, yline7/au, zline7/au, 'cyan', label='LOS 2')
# ax.plot3D(xline8/au, yline8/au, zline8/au, 'brown')
# ax.plot3D(xline9/au, yline9/au, zline9/au, 'yellow', label='LOS 3')
# ax.scatter3D(0.0, 0.0, 0.0, color='yellow', label='Sun')
# plt.legend()
# plt.xlabel('x (au)')
# plt.ylabel('y (au)')
# plt.title('Heliocentric orbit determination by Gauss method: Apophis')
# plt.show()
