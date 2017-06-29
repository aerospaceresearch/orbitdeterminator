import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from orbitdeterminator.filters import triple_moving_average as tma
import pytest

def test_weighted_average():
	assert tma.weighted_average([1, 1, 1]) == 1
	assert tma.weighted_average([0, 0, 0]) == 0
	assert tma.weighted_average([10]) == 10
	with pytest.raises(ZeroDivisionError):
		tma.weighted_average([])