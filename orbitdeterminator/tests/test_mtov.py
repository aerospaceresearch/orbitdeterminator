import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from orbitdeterminator.kep_determination import gibbs
import numpy as np
import pytest
from numpy.testing import assert_array_equal

# First test just checks the function with negative value for M
M = 350.0
e = 0.3
expected = gibbs.mtov(-10.0, e)


def test_mtov():
    assert_array_equal(gibbs.mtov(M, e), expected)

pytest.main()
