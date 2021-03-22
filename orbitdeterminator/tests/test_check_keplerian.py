import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from kep_determination import lamberts_kalman
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
def test_check_keplerian(given, expected):
    assert_array_equal(lamberts_kalman.check_keplerian(given), expected)


# Second test checks the function with three rows and the first and second row has wrong inputs (a < 0 or e < 0)
kep3 = np.array([[15700, -0.3, 90.0, 0.8, 0.0, 28],
                [-13700, 0.4, 90.0, 0.8, 0.2, 28],
                [13700, 0.4, 90.0, 0.8, 0.2, 28.]])


def test_check_keplerian_fail():
    with pytest.raises(AssertionError):
        assert_array_equal(lamberts_kalman.check_keplerian(kep3), kep3)


# Third test just the function with two rows and the second row has wrong inputs (e > 1)
kep4 = np.array([[20000, 0.3, 90.0, 0.8, 0.0, 28.0],
                [13741, 1.4, 90.0, 0.8, 0.2, 28.0]])


def test_check_keplerian_fail_2():
    with pytest.raises(AssertionError):
        assert_array_equal(lamberts_kalman.check_keplerian(kep4), kep4)
