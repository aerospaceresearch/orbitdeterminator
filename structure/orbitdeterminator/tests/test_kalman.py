import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.kep_determination import lamberts_kalman
import numpy as np
import pytest
from numpy.testing import assert_array_equal

# The first test checks the fact that if we give to the kalman filter three identical sets of keplerian elements
# the final approximation will be equal to this set
kep1 = np.array([[15711.578566, 0.377617, 90.0, 0.887383, 0.0, 28.35774],
				[15711.578566, 0.377617, 90.0, 0.887383, 0.0, 28.35774],
				[15711.578566, 0.377617, 90.0, 0.887383, 0.0, 28.35774]])


@pytest.mark.parametrize("givenkep1, givenR, expected", [
	(kep1, 0.01**2, kep1[0, :])
])
def test_simple_pass(givenkep1, givenR, expected):
	assert_array_equal(np.ravel(lamberts_kalman.kalman(givenkep1, givenR)), expected)


# The second test checks the fact that if we change the R parameter for the same given set it produces a different
# result
kep2 = np.array([[12311.578566, 0.177617, 90.0, 0.83247383, 0.0, 28.34574],
				[14511.578566, 0.277617, 90.2, 0.83247383, 0.1, 28.3543574],
				[15711.578566, 0.377617, 90.1, 0.8687383, 0.011, 28.6774]])

given = lamberts_kalman.kalman(kep2, 0.01**2)
expected = lamberts_kalman.kalman(kep2, 0.001**2)

@pytest.mark.xfail(reason= "We change the R parameter which changes the result")
def test_fail():
	assert_array_equal(given, expected)

pytest.main()
