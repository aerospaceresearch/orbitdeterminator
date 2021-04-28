"""SGP4 propagator. This is a wrapper around the PyPI SGP4 propagator.
   However, this does not generate an artificial TLE. So there is no
   string manipulation involved. Hence this is faster than sgp4_prop_string."""

from datetime import datetime
import numpy as np
from sgp4.model import Satellite
from sgp4.earth_gravity import wgs72
from sgp4.propagation import sgp4init
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util.state_kep import state_kep

def __true_to_mean(T,e):
    """Converts true anomaly to mean anomaly.

       Args:
           T(float): true anomaly in degrees
           e(float): eccentricity

       Returns:
           float: the mean anomaly in degrees
    """

    T = np.radians(T)
    E = np.arctan2((1-e**2)*np.sin(T),e+np.cos(T))
    M = E - e*np.sin(E)
    M = np.degrees(M)
    M = M%360
    return M

# Parts of this method have been copied from:
# https://github.com/brandon-rhodes/python-sgp4/blob/master/sgp4/io.py
def kep_to_sat(kep,epoch,bstar=0.21109E-4,whichconst=wgs72,afspc_mode=False):
    """kep_to_sat(kep,epoch,bstar=0.21109E-4,whichconst=wgs72,afspc_mode=False)

       Converts a set of keplerian elements into a Satellite object.

       Args:
           kep(1x6 numpy array): the osculating keplerian elements at epoch
           epoch(float): the epoch
           bstar(float): bstar drag coefficient
           whichconst(float): gravity model. refer pypi sgp4 documentation
           afspc_mode(boolean): refer pypi sgp4 documentation

      Returns:
           Satellite object: an sgp4 satellite object encapsulating the arguments
    """

    deg2rad  =  np.pi / 180.0;         #    0.0174532925199433
    xpdotp   =  1440.0 / (2.0 * np.pi);  #  229.1831180523293

    tumin = whichconst.tumin

    satrec = Satellite()
    satrec.error = 0;
    satrec.whichconst = whichconst # Python extension: remembers its consts

    satrec.satnum = 0
    dt_obj = datetime.utcfromtimestamp(epoch)
    t_obj = dt_obj.timetuple()
    satrec.epochdays = (t_obj.tm_yday +
                        t_obj.tm_hour/24 +
                        t_obj.tm_min/1440 +
                        t_obj.tm_sec/86400)
    satrec.ndot = 0
    satrec.nddot = 0
    satrec.bstar = bstar

    satrec.inclo = kep[2]
    satrec.nodeo = kep[4]
    satrec.ecco = kep[1]
    satrec.argpo = kep[3]
    satrec.mo = __true_to_mean(kep[5],kep[1])
    satrec.no = 86400/(2*np.pi*(kep[0]**3/398600.4405)**0.5)

    satrec.no   = satrec.no / xpdotp; #   rad/min
    satrec.a    = pow( satrec.no*tumin , (-2.0/3.0) );

    #  ---- find standard orbital elements ----
    satrec.inclo = satrec.inclo  * deg2rad;
    satrec.nodeo = satrec.nodeo  * deg2rad;
    satrec.argpo = satrec.argpo  * deg2rad;
    satrec.mo    = satrec.mo     * deg2rad;

    satrec.alta = satrec.a*(1.0 + satrec.ecco) - 1.0;
    satrec.altp = satrec.a*(1.0 - satrec.ecco) - 1.0;

    satrec.epochyr = dt_obj.year
    satrec.jdsatepoch = epoch/86400.0 + 2440587.5
    satrec.epoch = dt_obj

    #  ---------------- initialize the orbit at sgp4epoch -------------------
    sgp4init(whichconst, afspc_mode, satrec.satnum, satrec.jdsatepoch-2433281.5, satrec.bstar,
             satrec.ecco, satrec.argpo, satrec.inclo, satrec.mo, satrec.no,
             satrec.nodeo, satrec)

    return satrec

def propagate_kep(kep,t0,tf,bstar=0.21109E-4):
    """Propagates a set of keplerian elements.

       Args:
           kep(1x6 numpy array): osculating keplerian elements at epoch
           t0(float): initial time (epoch)
           tf(float): final time

       Returns:
           pos(1x3 numpy array): the position at tf
           vel(1x3 numpy array): the velocity at tf
    """

    sat = kep_to_sat(kep,t0,bstar=bstar)
    tf = datetime.utcfromtimestamp(tf).timetuple()
    pos, vel = sat.propagate(
        tf.tm_year, tf.tm_mon, tf.tm_mday, tf.tm_hour, tf.tm_min, tf.tm_sec)

    return np.array(list(pos)),np.array(list(vel))

def propagate_state(r,v,t0,tf,bstar=0.21109E-4):
    """Propagates a state vector

       Args:
           r(1x3 numpy array): the position vector at epoch
           v(1x3 numpy array): the velocity vector at epoch
           t0(float): initial time (epoch)
           tf(float): final time

       Returns:
           pos(1x3 numpy array): the position at tf
           vel(1x3 numpy array): the velocity at tf
    """

    kep = state_kep(r,v)
    return propagate_kep(kep,t0,tf,bstar)

if __name__ == "__main__":

    t0 = 1526927274
    tf = 1526932833

    #kep = np.array([6782.96, 0.0004084, 51.6402, 108.2140, 150.4026, 238.0528])

    r = np.array([-5.23684633e+03, 4.12417773e+03, -1.26294137e+03])
    v = np.array([-3.86204515e+00, -3.12048032e+00, 5.83839029e+00])

    #pos,vel = propagate_kep(kep,t0,tf)
    pos,vel = propagate_state(r,v,t0,tf)

    print(pos,vel)
