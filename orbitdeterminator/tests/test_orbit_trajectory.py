import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.kep_determination import lamberts_kalman
import numpy as np
import pytest
from numpy.testing import assert_array_equal


# This test checks the function by giving two points to it. In the first case we give the points as point1 and point2
# and it returns True because the motion is retrogade. Now if we put the points in reverse order we expect to get
# a false value meaning that the motion to be counter-clock wise.
x1 = np.array([0.0, 9590.0, -16.0, -80.0])
x2 = np.array([100.0, 9593.0, -35.0, 740.0])
x1_new = [1, 1, 1]
x1_new[:] = x1[1:4]
x2_new = [1, 1, 1]
x2_new[:] = x2[1:4]
time = x2[0] - x1[0]


@pytest.mark.parametrize("givenx1,givenx2,giventime, expected", [
    (x1_new, x2_new, time, True),
    (x2_new, x1_new, time, False)
])
def test_orbit_trajectory(givenx1, givenx2, giventime, expected):
    assert_array_equal(lamberts_kalman.orbit_trajectory(givenx1, givenx2, giventime), expected)
