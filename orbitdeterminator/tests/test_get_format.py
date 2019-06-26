import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from util import get_format as gf
import pytest

TEST_FILE = os.path.dirname(os.path.realpath(__file__)) + "/../example_data/SourceTXT/"

def test_get_format():
	assert gf.get_format(TEST_FILE + "uk_sample.txt") == "UK"
	assert gf.get_format(TEST_FILE + "rde_sample.txt") == "RDE"
	assert gf.get_format(TEST_FILE + "iod_sample.txt") == "IOD"
	assert gf.get_format(os.path.dirname(os.path.realpath(__file__)) + "/../example_data/SourceCSV/orbit.csv") == "Cartesian"