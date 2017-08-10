import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.kep_determination import (lamberts_kalman, interpolation)
import numpy as np
import pytest
from numpy.testing import assert_array_equal


@pytest.fixture()
def my_kep():
    kep = np.array([[100, 10100, 9500, 20500],
                    [200, 10500, 9200, 20700],
                    ], dtype=float)
    return kep


# Checks if the output lenght is 6 == keplerian elements number
def test_output_len(my_kep):
    assert len(np.ravel(lamberts_kalman.create_kep(my_kep))) == 6
