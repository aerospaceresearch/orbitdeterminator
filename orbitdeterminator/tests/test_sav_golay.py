import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from filters import sav_golay
import numpy as np
from numpy.testing import assert_array_equal
import pytest


@pytest.fixture
def my_data():
	data = np.array([[100, 1000, 2000, 3000],
				[200, 1500, 500, 2500],
				[300, 1700, 200, 2200],
				[400, 1800, 100, 2000],
				[500, 1900, 20, 1700]
	])
	return data


# Checks that a change in the window parameter of the golay filter changes the result
def test_golay_params(my_data):
	given = sav_golay.golay(my_data, 3, 1)
	expected = sav_golay.golay(my_data, 5, 1)
	with pytest.raises(AssertionError):
		assert_array_equal(given, expected)


# Checks that the input and output data are of the same length
def test_golay_length(my_data):
	given = len(my_data)
	expected = len(sav_golay.golay(my_data, 3, 1))
	assert given == expected


# Checks that if you input an odd integer for window lenght it raises an error
def test_golay_parmerrors(my_data):
	with pytest.raises(Exception):
		result = sav_golay.golay(my_data, 2, 1)


# Checks that if you input an integer for the polynomial parameter equal or larger than the window lenght ir raises
# an error
def test_golay_parmerrors(my_data):
	with pytest.raises(Exception):
		result = sav_golay.golay(my_data, 3, 3)
