import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from util import get_format as gf
import pytest

TEST_FILE = os.path.dirname(os.path.realpath(__file__)) + "/../example_data/Test/"

def test_get_format():
	assert gf.get_format(TEST_FILE + "test_uk.txt") == "UK"
	assert gf.get_format(TEST_FILE + "test_rde.txt") == "RDE"
	assert gf.get_format(TEST_FILE + "test_iod.txt") == "IOD"