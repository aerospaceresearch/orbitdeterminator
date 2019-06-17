import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util import get_format as gf
import pytest

def test_get_format():
	assert gf.get_format("example_data/Test/test_uk.txt") == "UK"
	assert gf.get_format("example_data/Test/test_rde.txt") == "RDE"
	assert gf.get_format("example_data/Test/test_iod.txt") == "IOD"