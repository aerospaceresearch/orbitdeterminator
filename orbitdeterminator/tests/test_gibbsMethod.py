import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pytest
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

def test_vector_sum():
    obj = Gibbs()

    vec1 = [123.76, 233.98, 675.21]
    vec2 = [13.234, 231.235, 8776.65]

    vec = obj.vector_sum(vec1, vec2, 1)
    vec = [float("{0:.2f}".format(i)) for i in vec]
    ans = [136.99, 465.22, 9451.86]
    assert_array_equal(vec, ans)

    vec = obj.vector_sum(vec1, vec2, 0)
    vec = [float("{0:.2f}".format(i)) for i in vec]
    ans = [110.53, 2.74, -8101.44]
    assert_array_equal(vec, ans)

    vec1 = [435.3452, -655.621, 956.075]
    vec2 = [537.956, 392.374, -755.343]

    vec = obj.vector_sum(vec1, vec2, 1)
    vec = [float("{0:.2f}".format(i)) for i in vec]
    ans = [973.3, -263.25, 200.73]
    assert_array_equal(vec, ans)

    vec = obj.vector_sum(vec1, vec2, 0)
    vec = [float("{0:.2f}".format(i)) for i in vec]
    ans = [-102.61, -1047.99, 1711.42]
    assert_array_equal(vec, ans)

    del(obj)

if __name__ == "__main__":
    test_convert_list()
    test_magnitude()
    test_dot_product()
    test_cross_product()
    test_vector_sum()
