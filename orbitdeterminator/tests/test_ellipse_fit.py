import pytest
import numpy as np
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from util.new_tle_kep_state import kep_to_state
from util.rkf5 import rkf5
from kep_determination.ellipse_fit import determine_kep

def test_ellipse_fit():
    #noaa-1
    kep = np.array([101.7540, 195.7370, 0.0031531, 352.8640, 117.2610, 12.53984625169364])
    r = kep_to_state(kep)
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
    kep = np.array([54.4058, 84.8417, 0.0142955, 74.4543, 193.5934, 2.00565117179872])
    r = kep_to_state(kep)
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
    kep = np.array([92.0287, 282.8216, 0.0005088, 298.0188, 62.0505, 14.52172969429489])
    r = kep_to_state(kep)
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
