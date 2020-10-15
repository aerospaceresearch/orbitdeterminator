import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import input_transf
import numpy as np
import pytest
from numpy.testing import assert_array_equal


# This test takes one array with cartesian coordinates transform it into spherical coordinates
# with cart_to_spher function and then apply to that computed array the vice versa process to
# see if the final results if the initial array

def test_imput_transf():
    expected = np.array([[0, 1000, 1000, 2000]])
    given = input_transf.cart_to_spher(expected)

    assert_array_equal(input_transf.spher_to_cart(given), expected)
