import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.kep_determination import lamberts_kalman
import numpy as np
import pytest
from numpy.testing import assert_array_equal

# First test just checks the function with two normal rows
kep1 = np.array([[15700, 0.3, 90.0, 0.8, 0.0, 28],
                [13700, 0.4, 90.0, 0.8, 0.2, 28]])

kep2 = np.array([[20000, 0.3, 90.0, 0.8, 0.0, 28],
                [13700, 0.4, 90.0, 0.8, 0.2, 28]])


@pytest.mark.parametrize("given, expected", [
    (kep1, kep1),
    (kep2, kep2),
])
def test_simple_pass(given, expected):
    assert_array_equal(lamberts_kalman.check_keplerian(given), expected)

kep1 = np.array([[15700, -0.3, 90.0, 0.8, 0.0, 28],
                [-13700, 0.4, 90.0, 0.8, 0.2, 28],
                [13700, 0.4, 90.0, 0.8, 0.2, 28.]])


# Second test checks the function with three rows that the first and second row has wrong inputs
@pytest.mark.xfail(reason="First row has negative eccentricity, Second row has negative semi major axis")
def test_fail(given, expected):
    assert_array_equal(lamberts_kalman.check_keplerian(kep1), kep1)


# Third test just the function with three rows that the first row has wrong inputs
kep2 = np.array([[20000, 0.3, 90.0, 0.8, 0.0, 28.0],
                [13741, 1.4, 90.0, 0.8, 0.2, 28.0]])


@pytest.mark.xfail(reason=" Second row has eccentricity > 1")
def test_fail_2():
    assert_array_equal(lamberts_kalman.check_keplerian(kep2), kep2)


pytest.main()
