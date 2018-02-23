import sys
import os
import pytest
import numpy as np
from orbitdeterminator.filters import smooth_moving_average as sma


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_weighted_average():
    assert sma.distance_weighted_average(np.array([1, 1])) == 1
    assert sma.distance_weighted_average(np.array([0, 0])) == 0
    assert sma.distance_weighted_average(np.array([1, 2])) == 4 / 3
    assert sma.distance_weighted_average(np.array([1, 2]), 'r') == 5 / 3
    assert sma.distance_weighted_average(np.array([10])) == 10
    with pytest.raises(ZeroDivisionError):
        sma.distance_weighted_average(np.array([]))


def main():
    test_weighted_average()


if __name__ == "__main__":
    main()
