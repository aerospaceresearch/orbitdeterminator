import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal
from kep_determination.gibbsMethod import *

def test_convert_list():
    obj = Gibbs()

    vec1 = ['12346.23434', '123.456', '456.789', '789.123']
    vec = obj.convert_list(vec1)
    assert(isinstance(vec[0], float) == True)
    assert(isinstance(vec[1], float) == True)
    assert(isinstance(vec[2], float) == True)

    vec1 = ['5644.23456', '2634.1234', '74653.34958', '685476.2345']
    vec = obj.convert_list(vec1)
    assert(isinstance(vec[0], float) == True)
    assert(isinstance(vec[1], float) == True)
    assert(isinstance(vec[2], float) == True)

    del(obj)

def test_magnitude():
    obj = Gibbs()

    vec = [-294.32, 4265.1, 5986.7]
    mag = obj.magnitude(vec)
    mag = float("{0:.1f}".format(mag))
    ans = 7356.5
    assert_almost_equal(mag, ans)

    vec = [-1365.5, 3637.6, 6346.8]
    mag = obj.magnitude(vec)
    mag = float("{0:.1f}".format(mag))
    ans = 7441.7
    assert_almost_equal(mag, ans)

    vec = [-2940.3, 2473.7, 6555.8]
    mag = obj.magnitude(vec)
    mag = float("{0:.1f}".format(mag))
    ans = 7598.9
    assert_almost_equal(mag, ans)

    del(obj)

def test_dot_product():
    obj = Gibbs()

    vec1 = [123.76, 233.98, 675.21]
    vec2 = [13.234, 231.235, 8776.65]
    dot = obj.dot_product(vec1, vec2)
    dot = float("{0:.2f}".format(dot))
    ans = 5981824.05
    assert_almost_equal(dot, ans)

    vec1 = [435.3452, -655.621, 956.075]
    vec2 = [537.956, 392.374, -755.343]
    dot = obj.dot_product(vec1, vec2)
    dot = float("{0:.2f}".format(dot))
    ans = -745216.63
    assert_almost_equal(dot, ans)

    del(obj)

def test_cross_product():
    obj = Gibbs()

    vec1 = [123.76, 233.98, 675.21]
    vec2 = [13.234, 231.235, 8776.65]
    cross = obj.cross_product(vec1, vec2)
    cross = [float("{0:.2f}".format(i)) for i in cross]
    ans = [1897428.38, -1077262.47, 25521.15]
    assert_array_equal(cross, ans)

    vec1 = [435.3452, -655.621, 956.075]
    vec2 = [537.956, 392.374, -755.343]
    cross = obj.cross_product(vec1, vec2)
    cross = [float("{0:.2f}".format(i)) for i in cross]
    ans = [120079.76, 843161.23, 523513.39]
    assert_array_equal(cross, ans)

    del(obj)

def test_operate_vector():
    obj = Gibbs()

    vec1 = [123.76, 233.98, 675.21]
    vec2 = [13.234, 231.235, 8776.65]

    vec = obj.operate_vector(vec1, vec2, 1)
    vec = [float("{0:.2f}".format(i)) for i in vec]
    ans = [136.99, 465.22, 9451.86]
    assert_array_equal(vec, ans)

    vec = obj.operate_vector(vec1, vec2, 0)
    vec = [float("{0:.2f}".format(i)) for i in vec]
    ans = [110.53, 2.74, -8101.44]
    assert_array_equal(vec, ans)

    vec1 = [435.3452, -655.621, 956.075]
    vec2 = [537.956, 392.374, -755.343]

    vec = obj.operate_vector(vec1, vec2, 1)
    vec = [float("{0:.2f}".format(i)) for i in vec]
    ans = [973.3, -263.25, 200.73]
    assert_array_equal(vec, ans)

    vec = obj.operate_vector(vec1, vec2, 0)
    vec = [float("{0:.2f}".format(i)) for i in vec]
    ans = [-102.61, -1047.99, 1711.42]
    assert_array_equal(vec, ans)

    del(obj)


def test_unit():
    obj = Gibbs()

    vec = [8.1473, -9.7095, 7.3179]
    unit = obj.unit(vec)
    unit = [float("{0:.5f}".format(i)) for i in unit]
    ans = [0.55667, -0.66341, 0.5]
    assert_array_equal(unit, ans)

    vec = [-294.32, 4265.1, 5986.7]
    unit = obj.unit(vec)
    unit = [float("{0:.5f}".format(i)) for i in unit]
    ans = [-0.04001, 0.57977, 0.8138]
    assert_array_equal(unit, ans)

    del(obj)

def test_gibbs():
    obj = Gibbs()

    r1 = [-294.32, 4265.1, 5986.7]
    r2 = [-1365.5, 3637.6, 6346.8]
    r3 = [-2940.3, 2473.7, 6555.8]
    v = obj.gibbs(r1, r2, r3)
    v = [float("{0:.4f}".format(i)) for i in v]
    ans = [-6.2174, -4.0122, 1.599]
    assert_array_equal(v, ans)

    del(obj)

def test_orbital_elements():
    obj = Gibbs()

    r = [-1365.5, 3637.6, 6346.8]
    v = [-6.2174, -4.0122, 1.599]
    ele = obj.orbital_elements(r, v)
    ele = [float("{0:.2f}".format(i)) for i in ele]
    # Old format not in consistence with main.py
    # ans = [8001.48, 60.0, 40.0, 0.1, 30.08, 49.92]
    # New format in consistence with main.py
    ans = [8001.48, 0.1, 60.0, 30.08, 40.0, 49.92]
    assert_array_equal(ele, ans)

    r = [-6045, -3490, 2500]
    v = [-3.457, 6.618, 2.533]
    ele = obj.orbital_elements(r, v)
    ele = [float("{0:.2f}".format(i)) for i in ele]
    # Old format not in consistence with main.py
    # ans = [8788.08, 153.25, 255.28, 0.17, 20.07, 28.45]
    # New format in consistence with main.py
    ans = [8788.08, 0.17, 153.25, 20.07, 255.28, 28.45]
    assert_array_equal(ele, ans)

    r = [5.0756899358316559e+03, -4.5590381308371752e+03, 1.9322228177731663e+03]
    v = [1.3360847905126974e+00, -1.5698574946888049e+00, -7.2117328822023676e+00]
    ele = obj.orbital_elements(r, v)
    ele = [float("{0:.2f}".format(i)) for i in ele]
    # Old format not in consistence with main.py
    # ans = [7096.68, 92.02, 137.5, 0.0, 159.0, 5.17]
    # New format in consistence with main.py
    ans = [7096.68, 0.0, 92.02, 159.0, 137.5, 5.17]
    assert_array_equal(ele, ans)

    del(obj)

if __name__ == "__main__":
    test_convert_list()
    test_magnitude()
    test_dot_product()
    test_cross_product()
    test_operate_vector()
    test_unit()
    test_gibbs()
    test_orbital_elements()
