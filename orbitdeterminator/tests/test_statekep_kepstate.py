import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import (state_kep, kep_state)
import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_array_equal


# Initial state vector which is going to be transformed into keplerian elements and then back to the initial state
kep = np.array([[15711.578566], [0.377617], [90.0], [0.887383], [0.0], [28.357744]])
y = kep_state.kep_state(kep)

expected = np.ravel(kep)
r_given = np.array([y[0, 0], y[1, 0], y[2, 0]])
v_given = np.array([y[3, 0], y[4, 0], y[5, 0]])


# Test the vice versa procedure with an uncertainty of 10 decimal points
def test_statekep_kepstate():
	npt.assert_almost_equal(state_kep.state_kep(r_given, v_given), expected, decimal=10)

