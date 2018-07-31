import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from numpy.testing import assert_array_equal
from kep_determination.sgp4 import *

def test_proapagation_model():
    obj = SGP4()

    line1 = "1 88888U          80275.98708465  .00073094  13844-3  66816-4 0     8"
    line2 = "2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518   105"
    obj.compute_necessary(line1, line2)
    tsince = 0
    r,v = obj.propagation_model(tsince)
    r = [float("{0:.5f}".format(i)) for i in r]
    v = [float("{0:.5f}".format(i)) for i in v]
    pos = [2328.97070, -5995.22083, 1719.97066]
    vel = [2.91207, -0.98342, -7.09082]
    assert_array_equal(r, pos)
    assert_array_equal(v, vel)
    # print(r,v)

    line1 = "1 32785U 08021C   18201.86927515  .00000199  00000-0  27157-4 0  9996"
    line2 = "2 32785  97.5464 212.4389 0011563 289.3405  70.6562 14.88147354554182"
    obj.compute_necessary(line1, line2)
    tsince = 0
    r,v = obj.propagation_model(tsince)
    r = [float("{0:.5f}".format(i)) for i in r]
    v = [float("{0:.5f}".format(i)) for i in v]
    pos = [-5892.75718, -3745.27150, 0.00265]
    vel = [-0.53231, 0.83713, 7.49348]
    assert_array_equal(r, pos)
    assert_array_equal(v, vel)
    # print(r,v)

    line1 = "1 27844U 03031E   18209.96155204  .00000018  00000-0  28062-4 0  9995"
    line2 = "2 27844  98.6862 218.0011 0008601 248.4534 111.5728 14.22124843782096"
    obj.compute_necessary(line1, line2)
    tsince = 0
    r,v = obj.propagation_model(tsince)
    r = [float("{0:.5f}".format(i)) for i in r]
    v = [float("{0:.5f}".format(i)) for i in v]
    pos = [-5674.78436, -4433.80110, 0.00928]
    vel = [-0.69002, 0.88593, 7.35498]
    assert_array_equal(r, pos)
    assert_array_equal(v, vel)
    # print(r,v)

    del(obj)

def test_recover_tle():
    obj = SGP4()

    pos = [2.32897070e+03, -5.99522083e+03, 1.71997066e+03]
    vel = [2.91207000e+00, -9.83420000e-01, -7.09082000e+00]
    tle = obj.recover_tle(pos, vel)
    line1 = '1 xxxxxc xxxxxccc xxxxx.xxxxxxxx _.xxxxxxxx _xxxxx_x _xxxxx_x x xxxxx'
    line2 = '2 xxxxx  72.8539 115.9623 0096686  59.4225 104.8919 16.03893203xxxxxx'
    # print(tle[0])
    # print(tle[1])
    assert(tle[0] == line1)
    assert(tle[1] == line2)

    del(obj)

if __name__ == "__main__":
    test_proapagation_model()
    test_recover_tle()
