import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.kep_determination import (gibbs, lamberts_kalman)
import numpy as np
import pytest
from numpy.testing import assert_array_equal

@pytest.fixture()
def my_kep():
	kep = np.array([[0, 100, 1000, 2000],
					[200, 300, 800, 2500],
					[400, 600, 750, 2700],
					], dtype=float)
	return kep


# Checks if the output lenght is 6 == keplerian elements number
def test_output_len(my_kep):
	assert len(gibbs.kep_gibbs(my_kep[0, 1:4], my_kep[1, 1:4], my_kep[2, 1:4])) == 6


# Checks if the orbit computed from Gibbs method is equal to the one computed from Lambert's solution. Produces an
# assertion error meaning the two computations are not equal
def test_compare_with_lamberts(my_kep):
	with pytest.raises(AssertionError):
		expected = lamberts_kalman.create_kep(my_kep)
		assert_array_equal(gibbs.kep_gibbs(my_kep[0, 1:4], my_kep[1, 1:4], my_kep[2, 1:4]), expected)
