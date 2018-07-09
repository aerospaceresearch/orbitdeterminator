"""Tests ellipse_fit with satellites. Compatible with pytest."""

import pytest
import numpy as np
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from util.new_tle_kep_state import tle_to_state
from util.rkf5 import rkf5
from kep_determination.ellipse_fit import determine_kep

def test_ellipse_fit():
    """Tests ellipse fit with 8 satellites:
       * NOAA-1
       * GPS-23
       * Cryosat-2
       * NOAA-15
       * NOAA-18
       * NOAA-19
       * MOLNIYA 2-10
       * ISS

       To add your own test copy the template, put the 2nd row of the TLE of the satellite
       in place of kep. In the rkf5 line put the final time and time step such that 700±200
       points are generated. Now, put the actual orbital parameters in the assert statements.

       Args:
           NIL

       Returns:
           NIL
    """

    #noaa-1
    tle = np.array([101.7540, 195.7370, 0.0031531, 352.8640, 117.2610, 12.53984625169364])
    r = tle_to_state(tle)
    _,vecs = rkf5(0,7200,10,r)
    r = np.reshape(r,(1,6))
    vecs = np.insert(vecs,0,r,axis=0)
    vecs = vecs[:,0:3]

    kep,_ = determine_kep(vecs)
    assert kep[0] == pytest.approx(7826.006538, 0.1)    # sma
    assert kep[1] == pytest.approx(0.0031531, 0.01)     # ecc
    assert kep[2] == pytest.approx(101.7540, 0.1)       # inc
    assert kep[3] == pytest.approx(352.8640, 1.0)       # argp
    assert kep[4] == pytest.approx(195.7370, 0.1)       # raan
    assert kep[5] == pytest.approx(117.2610, 0.5)       # true_anom

    #gps-23
    tle = np.array([54.4058, 84.8417, 0.0142955, 74.4543, 193.5934, 2.00565117179872])
    r = tle_to_state(tle)
    _,vecs = rkf5(0,43080,50,r)
    r = np.reshape(r,(1,6))
    vecs = np.insert(vecs,0,r,axis=0)
    vecs = vecs[:,0:3]

    kep,_ = determine_kep(vecs)
    assert kep[0] == pytest.approx(26560.21419, 0.1)    # sma
    assert kep[1] == pytest.approx(0.0142955, 0.01)     # ecc
    assert kep[2] == pytest.approx(54.4058, 0.1)        # inc
    assert kep[3] == pytest.approx(74.4543, 1.0)        # argp
    assert kep[4] == pytest.approx(84.8417, 0.1)        # raan
    assert kep[5] == pytest.approx(193.5934, 0.5)       # true_anom

    #cryosat-2
    tle = np.array([92.0287, 282.8216, 0.0005088, 298.0188, 62.0505, 14.52172969429489])
    r = tle_to_state(tle)
    _,vecs = rkf5(0,5950,10,r)
    r = np.reshape(r,(1,6))
    vecs = np.insert(vecs,0,r,axis=0)
    vecs = vecs[:,0:3]

    kep,_ = determine_kep(vecs)
    assert kep[0] == pytest.approx(7096.69719, 0.1)     # sma
    assert kep[1] == pytest.approx(0.0005088, 0.01)     # ecc
    assert kep[2] == pytest.approx(92.0287, 0.1)        # inc
    assert kep[3] == pytest.approx(298.0188, 1.0)       # argp
    assert kep[4] == pytest.approx(282.8216, 0.1)       # raan
    assert kep[5] == pytest.approx(62.0505, 0.5)        # true_anom

    #noaa-15
    tle = np.array([98.7705, 158.2195, 0.0009478, 307.8085, 52.2235, 14.25852803])
    r = tle_to_state(tle)
    _,vecs = rkf5(0,6120,10,r)
    r = np.reshape(r,(1,6))
    vecs = np.insert(vecs,0,r,axis=0)
    vecs = vecs[:,0:3]

    kep,_ = determine_kep(vecs)
    assert kep[0] == pytest.approx(7183.76381, 0.1)     # sma
    assert kep[1] == pytest.approx(0.0009478, 0.01)     # ecc
    assert kep[2] == pytest.approx(98.7705, 0.1)        # inc
    assert kep[3] == pytest.approx(307.8085, 1.0)       # argp
    assert kep[4] == pytest.approx(158.2195, 0.1)       # raan
    assert kep[5] == pytest.approx(52.2235, 0.5)        # true_anom

    #noaa-18
    tle = np.array([99.1472, 176.6654, 0.0014092, 197.4778, 162.5909, 14.12376102669957])
    r = tle_to_state(tle)
    _,vecs = rkf5(0,6120,10,r)
    r = np.reshape(r,(1,6))
    vecs = np.insert(vecs,0,r,axis=0)
    vecs = vecs[:,0:3]

    kep,_ = determine_kep(vecs)
    assert kep[0] == pytest.approx(7229.38911, 0.1)     # sma
    assert kep[1] == pytest.approx(0.0014092, 0.01)     # ecc
    assert kep[2] == pytest.approx(99.1472, 0.1)        # inc
    assert kep[3] == pytest.approx(197.4778, 1.0)       # argp
    assert kep[4] == pytest.approx(176.6654, 0.1)       # raan
    assert kep[5] == pytest.approx(162.5909, 0.5)       # true_anom

    #noaa-19
    tle = np.array([99.1401, 119.3629, 0.0014753, 44.0001, 316.2341, 14.12279464478196])
    r = tle_to_state(tle)
    _,vecs = rkf5(0,6120,10,r)
    r = np.reshape(r,(1,6))
    vecs = np.insert(vecs,0,r,axis=0)
    vecs = vecs[:,0:3]

    kep,_ = determine_kep(vecs)
    assert kep[0] == pytest.approx(7229.71889, 0.1)    # sma
    assert kep[1] == pytest.approx(0.0014753, 0.01)    # ecc
    assert kep[2] == pytest.approx(99.1401, 0.1)       # inc
    assert kep[3] == pytest.approx(44.0001, 1.0)       # argp
    assert kep[4] == pytest.approx(119.3629, 0.1)      # raan
    assert kep[5] == pytest.approx(316.2341, 0.5)      # true_anom

    #molniya 2-10
    tle = np.array([63.2749, 254.2968, 0.7151443, 294.4926, 9.2905, 2.01190064320534])
    r = tle_to_state(tle)
    _,vecs = rkf5(0,43000,50,r)
    r = np.reshape(r,(1,6))
    vecs = np.insert(vecs,0,r,axis=0)
    vecs = vecs[:,0:3]

    kep,_ = determine_kep(vecs)
    assert kep[0] == pytest.approx(26505.1836, 0.1)     # sma
    assert kep[1] == pytest.approx(0.7151443, 0.01)     # ecc
    assert kep[2] == pytest.approx(63.2749, 0.1)        # inc
    assert kep[3] == pytest.approx(294.4926, 1.0)       # argp
    assert kep[4] == pytest.approx(254.2968, 0.1)       # raan
    assert kep[5] == pytest.approx(65.56742, 0.5)       # true_anom

    #ISS
    tle = np.array([51.6402, 150.4026, 0.0004084, 108.2140, 238.0528, 15.54082454114406])
    r = tle_to_state(tle)
    _,vecs = rkf5(0,5560,10,r)
    r = np.reshape(r,(1,6))
    vecs = np.insert(vecs,0,r,axis=0)
    vecs = vecs[:,0:3]

    kep,_ = determine_kep(vecs)
    assert kep[0] == pytest.approx(6782.95812, 0.1)     # sma
    assert kep[1] == pytest.approx(0.0004084, 0.01)     # ecc
    assert kep[2] == pytest.approx(51.6402, 0.1)        # inc
    assert kep[3] == pytest.approx(108.2140, 1.0)       # argp
    assert kep[4] == pytest.approx(150.4026, 0.1)       # raan
    assert kep[5] == pytest.approx(238.0528, 0.5)       # true_anom
