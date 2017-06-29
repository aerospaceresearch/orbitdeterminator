import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from orbitdeterminator.filters import triple_moving_average as tma

def test_weighted_average():
	avg = tma.weighted_average([1, 1, 1])
	assert avg == 1	 
