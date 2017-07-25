import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.kep_determination import lamberts_kalman
import numpy as np
import pytest
from numpy.testing import assert_array_equal

# The first test checks the fact that if we give to the kalman filter three identical sets of keplerian elements
# the final approximation will be equal to this set


def test_kalman():
    kep1 = np.array([[10000, 0.10, 90.0, 0.80, 0.0, 28.00],
                     [10000, 0.10, 90.0, 0.80, 0.0, 28.00],
                     [10000, 0.10, 90.0, 0.80, 0.0, 28.00]])

    assert_array_equal(np.ravel(lamberts_kalman.kalman(kep1, 0.01**2)), kep1[0, :])


# The second test checks the fact that if we change the R parameter for the same given set it produces a different
# result


def test_kalman_fail():
    kep2 = np.array([[10000, 0.10, 90.0, 0.80, 0.0, 28.00],
                     [12000, 0.20, 92.0, 0.82, 0.1, 28.20],
                     [13500, 0.15, 95.0, 0.85, 0.3, 28.50]])

    given = lamberts_kalman.kalman(kep2, 0.01 ** 2)
    expected = lamberts_kalman.kalman(kep2, 0.001 ** 2)

    with pytest.raises(AssertionError):
        assert_array_equal(given, expected)
