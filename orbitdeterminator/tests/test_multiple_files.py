import pytest
import numpy as np
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from kep_determination.ellipse_fit import determine_kep
from util import handle_multiple_files

def test_multiple_files():
	file_list1 = ["orbit_split1.csv", "orbit_uk_split2.txt"]
	file_list2 = ["orbit_uk_split1.txt", "orbit_split2.csv"]

	# Test1 with file_list1
	data = handle_multiple_files.handle_multiple_files(file_list1)
	kep, _ = determine_kep(data[:, 1:])
	assert kep[0] == pytest.approx(7268.79110357, 0.1)
	assert kep[1] == pytest.approx(0.23813232, 0.01)
	assert kep[2] == pytest.approx(28.98373017, 0.1)
	assert kep[3] == pytest.approx(273.00070838, 1.0)
	assert kep[4] == pytest.approx(336.14437702, 0.1)
	assert kep[5] == pytest.approx(308.45673089, 0.5)

	#Test2 with file_list2
	data = handle_multiple_files.handle_multiple_files(file_list2)
	kep, _ = determine_kep(data[:, 1:])
	assert kep[0] == pytest.approx(7268.79110357, 0.1)
	assert kep[1] == pytest.approx(0.23813232, 0.01)
	assert kep[2] == pytest.approx(28.98373017, 0.1)
	assert kep[3] == pytest.approx(273.00070838, 1.0)
	assert kep[4] == pytest.approx(336.14437702, 0.1)
	assert kep[5] == pytest.approx(308.45673089, 0.5)