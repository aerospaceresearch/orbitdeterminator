import numpy as np

from scipy.integrate import odeint
from sgp4.api import Satrec
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, ITRS, ICRS, TEME, CartesianDifferential, CartesianRepresentation

from orbdet.utils.constants import *
from orbdet.utils.utils import *

def get_satellite_sgp4(tle, epoch_start, epoch_end, step):
    """ Auxiliary function to obtain SGP4-propagated satellite coordinates
        within the specified epoch.

        # TODO: Different start/end Julian Days

    Args:
        tle (array): two-line element string array.
        epoch_start (astropy.time.Time): starting epoch.
        epoch_end (astropy.time.Time): ending epich.
        step (float): step in Julian Day fractions.
        verbose (bool): debug output. Defaults to False.

    Returns:
        e (np.ndarray): vector of SGP4 error codes.
        r (np.ndarray): vector of satellite positions (TEME).
        v (np.ndarray): vector of satellite velocities (TEME).
        jd (np.ndarray): vector of Julian Day numbers.
        fr (np.ndarray): vector of Julian Day fractions.
    """

    satellite = Satrec.twoline2rv(tle[0], tle[1])

    fr = np.arange(epoch_start.jd2, epoch_end.jd2, step)
    jd = np.ones(fr.shape[0]) * epoch_start.jd1

    e, r, v = satellite.sgp4_array(jd, fr)

    return e, r, v, jd, fr

def get_satellite(tle, epoch_start, epoch_end, step, frame='itrs'):
    """ Auxiliary function to get satellite coordinates in the specified frame 
        (ITRS or TEME), propagated using SGP with given Two-Line Element (TLE).
        Coordinates are returned as numpy array.
        
    Args:
        tle (array): two-line element string array.
        epoch_start (astropy.time.Time): starting epoch.
        epoch_end (astropy.time.Time): ending epich.
        step (float): step in Julian Day fractions.
        verbose (bool): debug output. Defaults to False.
        frame (str): frame (teme or itrs). Defaults to 'teme'.
    Returns:
        itrs    (astropy.coordinates.builtin_frames.itrs.ITRS): satellite position in ITRS.
        t       (astropy.time.core.Time): corresponding times
    """

    _, r, v, jd, fr = get_satellite_sgp4(tle, epoch_start, epoch_end, 1.0/86400.0)
    t = Time(jd + fr, format='jd')

    r_teme = CartesianRepresentation(r[:,0], r[:,1], r[:,2], unit=u.km)
    v_teme = CartesianDifferential(v[:,0], v[:,1], v[:,2], unit=u.km/u.s)
    teme = TEME(r_teme.with_differentials(v_teme), obstime=t)

    if frame=='teme':
        x_sat = np.array([teme.x.value, teme.y.value, teme.z.value, 
                                teme.v_x.value, teme.v_y.value, teme.v_z.value])
    elif frame=='itrs':
        itrs = teme.transform_to(ITRS(obstime=t))
        x_sat = np.array([itrs.x.value, itrs.y.value, itrs.z.value, 
                                itrs.v_x.value, itrs.v_y.value, itrs.v_z.value])

    return x_sat, t

def get_site(lat, lon, height, obstime, frame='teme'):
    """ Auxiliary function to obtain site coordinates in ITRS or TEME frame.

    Args:
        lat (float): latitude (degrees).
        lon (float): longitude (degrees).
        height (float): altitude (m).
        obstime (astropy.time.Time): time array (n, ).
        frame (str): frame (teme or itrs). Defaults to 'teme'.
    Returns:
        x_obs (np.ndarray): array with site positions in ITRS/TEME frame (6, n).
    """

    v = np.zeros(obstime.shape[0])      # Temporary variable

    if frame == 'itrs':
        site = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)
        site_itrs_temp = site.get_itrs(obstime=obstime)

        x_obs = np.array([site_itrs_temp.x.value, site_itrs_temp.y.value, site_itrs_temp.z.value,
            v, v, v])
            
    elif frame == 'teme':
        # Need some workaround conversions for TEME frame
        site = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height/1e3*u.km)
        site_itrs_temp = site.get_itrs(obstime=obstime)

        r_itrs = CartesianRepresentation(
            site_itrs_temp.data.xyz.value[0,:], 
            site_itrs_temp.data.xyz.value[1,:], 
            site_itrs_temp.data.xyz.value[2,:], unit=u.km)
        v_itrs = CartesianDifferential(v, v, v, unit=u.km/u.s)

        site_itrs = ITRS(r_itrs.with_differentials(v_itrs), obstime=obstime)

        site_teme = site_itrs.transform_to(TEME(obstime=obstime))
        x_obs = np.array([site_teme.x.value, site_teme.y.value, site_teme.z.value,
            site_teme.v_x.value, site_teme.v_y.value, site_teme.v_z.value])*1e3     # Meters

    return x_obs


def get_x_sat_odeint_stm(x_0, t):
    """ Auxiliary function to get odeint propagations of state vector and state transition matrix.
        
    Args:
        x_0 (np.ndarray): initial conditions (6, 1).
        t (np.ndarray): time array (n,).
    Returns:
        x_sat_orbdyn_stm (np.ndarray): odeint propagated position of the satellite (6, n).
        Phi (np.ndarray): array of corresponding state transition matrices (6, 6, n).
    """
    
    x_Phi_0 = np.concatenate([x_0.squeeze(), np.eye(x_0.shape[0]).flatten()])
    x_Phi = np.transpose(odeint(orbdyn_2body_stm, x_Phi_0, t, args=(MU,)))
    x_sat_orbdyn_stm = x_Phi[0:6,]
    Phi = x_Phi[6:,].reshape((x_0.shape[0], x_0.shape[0],  t.shape[0])) 

    return x_sat_orbdyn_stm, Phi

def get_example_scenario(id=0, frame='teme'):
    """ Auxiliary function to obtain example scenario variables. 
        Scenario 1 or 2 works.

    Args:
        id (int): Scenario id.
        frame (str): frame (teme or itrs). Defaults to 'teme'.
    Returns:
        x_0 (np.ndarray): initial satellite position in ITRF frame.
        t_sec (np.ndarray): time array (seconds).
        x_sat_orbdyn_stm (np.ndarray): odeint propagated position of the satellite.
        x_obs_1 (np.ndarray): observer 1 position.
        x_obs_multiple (np.ndarray): multiple observer positions.
        f_downlink (float): downlink frequency of the satellite.
    """
    f_downlink = [435.103, 145.980, 137.620]
    epoch_start = [Time('2020-05-27 23:46:00'), Time('2020-06-25 06:28:00'), Time('2020-07-02 22:45:00')]
    epoch_end   = [Time('2020-05-27 23:50:00'), Time('2020-06-25 06:38:00'), Time('2020-07-02 23:55:00')]

    tle = dict.fromkeys(range(3), [])
    # Scenario 0 - FALCONSAT-3, Sites: Atlanta, Jacksonville, Charlotte
    tle[0] = [  '1 30776U 07006E   20146.24591950  .00002116  00000-0  57170-4 0  9998',
                '2 30776  35.4350  68.4822 0003223 313.1473  46.8985 15.37715972733265']
    # Scenario 1 - FOX-1A (AO-85), Sites: Santiago, La Serena, ~La Silla
    tle[1] = [  '1 40967U 15058D   20175.33659500 +.00000007 +00000+0 +20124-4 0   687',
                '2 40967  64.7742 112.9087 0170632  72.3744 289.5913 14.76130447162443']
    # Scenario 1 - FOX-1A (AO-85), Sites: Santiago, La Serena, ~La Silla
    tle[2] = [  '1 25544U 98067A   20176.97949255  .00000516  00000-0  17286-4 0  9996',
                '2 25544  51.6446 309.8972 0002538  80.4322  66.5560 15.49457702233227']
    
    x_sat, t = get_satellite(tle[id], epoch_start[id], epoch_end[id], 1.0/86400.0, frame=frame)

    # Set first position
    x_0 = np.expand_dims(x_sat[:,0] * 1e3, axis=1)
    t_sec = t.to_value('unix')
    t_sec -= t_sec[0]

    # Propagate in order to get range rate measurements
    x_sat_orbdyn_stm, _ = get_x_sat_odeint_stm(x_0, t_sec)
    
    # Set observer position
    if id==0:
        x_obs_1 = get_site(33.7743331, -84.3970209, 288, obstime=t, frame=frame)   # Atlanta
        x_obs_2 = get_site(30.3449153, -81.8231881, 100, obstime=t, frame=frame)   # Jacksonville
        x_obs_3 = get_site(35.2030728, -80.9799098, 100, obstime=t, frame=frame)   # Charlotte
        
        x_obs_4 = get_site(36.1755204, -86.8595446, 100, obstime=t, frame=frame)   # Test

        #x_obs_multiple = np.transpose(np.concatenate([[x_obs_1], [x_obs_2]]), (1,2,0))
        x_obs_multiple = np.transpose(np.concatenate([[x_obs_1], [x_obs_2], [x_obs_3]]), (1,2,0))
        #x_obs_multiple = np.transpose(np.concatenate([[x_obs_1], [x_obs_2], [x_obs_3], [x_obs_4]]), (1,2,0))

    elif id==1:
        x_obs_1 = get_site(-33.43, -70.61, 500, obstime=t, frame=frame)   # Santiago
        x_obs_2 = get_site(-30.02, -70.70, 700, obstime=t, frame=frame)   # Vicuna
        x_obs_3 = get_site(-28.92, -70.58, 2000, obstime=t, frame=frame)   # ~La Silla
        x_obs_multiple = np.transpose(np.concatenate([[x_obs_1], [x_obs_2], [x_obs_3]]), (1,2,0))

    elif id==2:
        # TODO: Fix 
        x_obs_1 = get_site(51.1483578, -1.4384458, 100, obstime=t, frame=frame)   # Santiago
        x_obs_2 = get_site(44.075, 5.5346, 50, obstime=t, frame=frame)   # Vicuna
        x_obs_3 = get_site(48.835, 2.280, 50, obstime=t, frame=frame)   # ~La Silla
    #x_obs_multiple = np.transpose(np.concatenate([[x_obs_1], [x_obs_2], [x_obs_3]]), (1,2,0))
    
    return x_0, t_sec, x_sat_orbdyn_stm, x_obs_multiple, f_downlink[id]