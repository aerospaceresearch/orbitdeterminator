import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.kep_determination import lamberts_kalman
import numpy as np
import pytest
from numpy.testing import assert_array_equal

## First test just checks the normal functionality with two normal rows
kep1 = np.array([[15711.578566, 0.377617, 90.0, 0.887383, 0.0, 28.357744],
				[13741.575236, 0.477617, 90.0, 0.853233, 0.2, 28.332744]])

kep2 = np.array([[2000711.578566, 0.377617, 90.0, 0.887383, 0.0, 28.357744],
				[13741.575236, 0.477617, 90.0, 0.853233, 0.2, 28.332744]])

@pytest.mark.parametrize("given, expected", [
	(kep1, kep1),
	(kep2, kep2),
])

def test_simple_pass(given, expected):
	assert_array_equal(lamberts_kalman.check_keplerian(given), expected)

kep1 = np.array([[15711.578566, -0.377617, 90.0, 0.887383, 0.0, 28.35774],
				[-13741.575236, 0.477617, 90.0, 0.853233, 0.2, 28.332744],
				[13741.575236, 0.477617, 90.0, 0.853233, 0.2, 28.332744]])

## Second test just checks the normal functionality with three rows that the first and second rows has wrong inputs
@pytest.mark.xfail(reason="First row has negative eccentricity, Second row has negative semi major axis")
def test_fail(given, expected):
	assert_array_equal(lamberts_kalman.check_keplerian(kep1), kep1)


## Third test just checks the normal functionality with three rows that the first row has wrong inputs
kep2 = np.array([[2000711.578566, 0.377617, 90.0, 0.887383, 0.0, 28.357744],
				[13741.575236, 1.477617, 90.0, 0.853233, 0.2, 28.332744]])

@pytest.mark.xfail(reason=" Second row has eccentricity > 1")
def test_fail_2():
	assert_array_equal(lamberts_kalman.check_keplerian(kep2), kep2)


pytest.main()
