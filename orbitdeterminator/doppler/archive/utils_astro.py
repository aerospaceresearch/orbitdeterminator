import numpy as np
from math import fmod, pi, floor, sqrt

from orbdet.utils.constants import *

# def get_jd(date):
#     """ Obtain Julian Day given date.

#     Args:
#         date (datetime): date to be converted
#     Returns
#         jd (float): julian day number 
#     """
#     y = date[0]
#     m = date[1]

#     if m == 1 or m == 2:
#         m = m + 12
#         y = y - 1

#     t = floor(y * 0.01)
#     b = 2 - t + floor(t * 0.25)
#     c = ((date[5] / 60.0 + date[4]) / 60.0 + date[3]) / 24.0

#     jd = floor(365.25 * (y + 4716.0)) + floor(30.6001 * (m + 1)) \
#         + date[2] + b - 1524.5 + c

#     return jd

def get_jd(date):
    """ Obtain Julian Day given date.

    Args:
        date (datetime): date to be converted
    Returns
        jd (float): julian day number 
    """

    jd = 367.0 * date[0]  \
        - np.floor( (7 * (date[0] + floor( (date[1] + 9) / 12.0) ) ) * 0.25 ) \
        + np.floor( 275 * date[1] / 9.0 ) + date[2] + 1721013.5
    jdfrac = (date[5] + date[4] * 60.0 + date[3] * 3600.0) / 86400.0

    if jdfrac > 1.0:
        jd = jd + np.floor(jdfrac)
        jdfrac = jdfrac - np.floor(jdfrac)
        
    return jd, jdfrac

def get_ttt(jd:float):
    """ Get Julian centuries.
        
    Args:
        jd (float): Julan day number.
    Returns:
           (float): Julian centuries.
    """
    return (jd - 2451545.0) / 36525.0

def get_gmst(jd_ut_1):
    """ Obtain Greenwitch Mean Sidereal Time (GMST).
    
    Args:
        jjd_ut_1d (float):Julian Day Number
    Returns:
        theta_gmst (float): Greenwitch Mean Sidereal Time
    """

    t_ut_1 = (jd_ut_1 - 2451545.0) / 36525.0
    theta_gmst = -6.2e-6 * t_ut_1**3 + 0.093104 * t_ut_1**2 \
        + (876600.0 * 3600.0 + 8640184.812866) * t_ut_1 + 67310.54841
    theta_gmst = fmod(theta_gmst / 240.0 * pi / 180.0, 2.0*pi)

    if theta_gmst < 0.0:
        theta_gmst = theta_gmst + 2.0*pi

    return theta_gmst

# def eci_to_ecef(eci, jd):
#     """ Earth-Centered Inertial (ECI) to Earth-Centered, Earth-Fixed (ECEF)
#     coordinate system conversion.
    
#     TODO: Vectorize

#     Args:
#         eci (np.array): Vector in ECI coordinate system (pos, velocity)
#         jd (float): Julian Day Number
#     Returns:
#         ecef (np.array): Vector in ECEF coordinate system
#     """
#     gmst = get_gmst(jd)
#     rotGMST = rot_3(gmst)
#     ecef_pos = rotGMST.dot(eci[0:3])
#     ecef_vel = rotGMST.dot(eci[3:6]) - np.cross(OMEGA_EARTH, ecef_pos)
#     ecef = np.concatenate((ecef_pos, ecef_vel), axis=0)

#     return ecef

def ecef_to_pef(ecef: np.array, ttt:float, xp: float, yp: float):
    """ Earth-Centered, Earth-Fixed (ECEF) to Pseudo Earth-Fixed (PEF) frame.

    TODO: Vectorize

    Args: 
        ecef (np.ndarray): ECEF coordinates
        jd (float): Julian day number
        ttt (np.ndarray): Julian centures
        xp: (float): x polar motion coefficient. Defaults to 0.
        yp: (float): y polar motion coefficient. Defaults to 0.
    Returns:
        pef (np.array): PEF coordinates
    """

    pm = polar_motion(xp, yp, ttt)

    r_pef = np.dot(pm, ecef[0:3])
    v_pef = np.dot(pm, ecef[3:6])

    pef = np.concatenate((r_pef, v_pef))

    return pef

def ecef_to_teme(ecef: np.array, jd: float, t_tt: float, lod: float, xp: float, yp: float, eqeterms: int):
    """ Earth-Centered, Earth-Fixed (ECEF) to True Equator, Mean Equinox frame
    coordinate system conversion. TEME frame is used for the NORAD two-line elements.

    TODO: Vectorize

    Args: 
        ecef (np.ndarray): ECEF coordinates
        t_tt (np.ndarray): Julian centures
        jd (float): Julian day number
        lod: (float): Excess length of day. Defaults to 0.
        xp: (float): x polar motion coefficient. Defaults to 0.
        yp: (float): y polar motion coefficient. Defaults to 0.
        eqterms (int): extra kinematic terms usage (after 1997). Options: [0, 2]. Defaults to 0.
    Returns:
        teme (np.array): TEME coordinates
    """

    gmst = get_gmst(jd)

    omega = 125.04452222 + (-6962890.5390*t_tt + 7.455*t_tt**2 + 0.008*t_tt**3)  / 3600.0
    omega = np.deg2rad(fmod(omega, 360.0))

    if jd > 2450449.5 and eqeterms > 0:
        gmst = gmst + 0.00264*pi/(3600.0*180.0)*np.sin(omega) \
            + 0.000063*pi/(3600.0*180.0)*np.sin(2.0*omega)

    gmst = fmod(gmst, 2.0*pi)
    st = rot_3(-gmst)

    pef = ecef_to_pef(ecef, t_tt, xp, yp)

    r_teme = np.dot(st, pef[0:3])
    v_teme = np.dot(st, pef[3:6] + np.cross(OMEGA_EARTH, pef[0:3]))

    teme = np.concatenate((r_teme, v_teme))

    # TODO: Add acceleration ?

    return teme

# # ECEF to ECI
# def ecef_to_eci(ecef, jd):
#     """ Earth-Centered, Earth-Fixed (ECEF) to Earth-Centered Inertial (ECI)
#     coordinate system conversion.
    
#     TODO: Vectorize

#     Args:
#         eci (np.array): Vector in ECI coordinate system (pos, velocity)
#         jd (float): Julian Day Number
#     Returns:
#         ecef (np.array): Vector in ECEF coordinate system
#     """
#     rot_gmst = rot_z(get_gmst(jd))
#     eci_pos = rot_gmst.dot(ecef[0:3])
#     eci_vel = rot_gmst.dot(ecef[3:6] + np.cross(OMEGA_EARTH, ecef[0:3]))
    
#     return np.concatenate((eci_pos, eci_vel), axis=0)

def geodetic_to_ecef(geo):
    """ Geodetic to Earth-Centered, Earth-Fixed coordinate System (ECEF).
    WGS-84 Standard (TODO: Verify).

    Args:
        geo (np.array): geodetic coordinates 
                        (latitude (rad), longitude (rad), altitude (m))
    Returns:
        ecef (np.array): vector in ECEF coordinates
    """
    
    s = np.sin(geo[0])
    N = R_EQ / sqrt(1.0 - E2 * pow(s, 2))
    t = (N + geo[2]) * np.cos(geo[0])
    
    ecef = np.array([t * np.cos(geo[1]), t * np.sin(geo[1]), ((1 - E2) * N + geo[2]) * s, 0, 0, 0])
    return ecef

def rot_x(a):
    """ Rotation around x axis.

    Args: 
        a (float): angle.
    Returns:
        r (np.array): rotation matrix
    """ 

    s,c = np.sin(a), np.cos(a)
    r = np.array([[1,0,0], [0,c,-s], [0,s,c]])
    return r

def rot_1(a):
    """ Rotation around x axis.

    Args: 
        a (float): angle.
    Returns:
        r (np.array): rotation matrix
    """ 

    s,c = np.sin(a), np.cos(a)
    r = np.array([[1,0,0], [0,c,s], [0,-s,c]])
    return r

def rot_y(a):
    """ Rotation around y axis.

    Args: 
        a (float): angle.
    Returns:
        r (np.array): rotation matrix
    """
    s,c = np.sin(a), np.cos(a)
    r = np.array([[c,0,s], [0,1,0], [-s,0,c]])
    return r

def rot_2(a):
    """ Rotation around y axis.

    Args: 
        a (float): angle.
    Returns:
        r (np.array): rotation matrix
    """
    s,c = np.sin(a), np.cos(a)
    r = np.array([[c,0,-s], [0,1,0], [s,0,c]])
    return r

def rot_z(a):
    """ Rotation around z axis.

    Args: 
        a (float): angle.
    Returns:
        r (np.array): rotation matrix
    """
    s,c = np.sin(a), np.cos(a)
    r = np.array([[c,-s,0], [s,c,0], [0,0,1]])
    return r

def rot_3(a):
    """ Rotation around z axis.

    Args: 
        a (float): angle.
    Returns:
        r (np.array): rotation matrix
    """
    s,c = np.sin(a), np.cos(a)
    r = np.array([[c,s,0], [-s,c,0], [0,0,1]])
    return r

def polar_motion(xp:float, yp:float, ttt:float=0.0, type:str='iau-76'):
    """ Polar motion rotation matrix.

    Args:
        xp (float): x polar motion coefficient (rad)
        yp (float): y polar motion coefficient (rad)
        ttt (float): Julian centrues of TT, IAU-2006
        type (str)
    Returns:
        pm (np.array): polar motion rotation matrix
    """

    if type=='iau-76':
        pm = np.dot(rot_1(yp), rot_2(xp))
    elif type=='iau-2000':
        sp = -47.0e-6 * ttt * pi / (3600.0 * 180.0)
        pm = np.linalg.multi_dot([rot_3(-sp), rot_1(yp), rot_2(xp)])
    
    return pm

