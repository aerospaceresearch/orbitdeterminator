import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import read_data
from math import *

def cart_to_spher(data):
    '''Takes as an input a data set containing points in cartesian format (time, x, y, z) and returns the computed
    spherical coordinates (time, azimuth, elevation, r)

    Args:
        data(numpy array) = containing the cartesian coordinates in format of (time, x, y, z)

    Returns:
        result(numpy array) = containing the spherical coordinates in format of (time, azimuth, elevation, r)
    '''

    for i in range(0, len(data)):
        x = data[i, 1]
        y = data[i, 2]
        z = data[i, 3]

        r = sqrt(x**2 + y**2 + z**2)
        elevation = atan2(z, sqrt(x**2 + y**2))
        azimuth = atan2(y, x)
        result = data
        result[i, 1] = azimuth
        result[i, 2] = elevation
        result[i, 3] = r

    return result


def spher_to_cart(data):
    ''' Takes as an input a data set containing points in spherical format (time, azimuth, elevation, r) and
    returns the computed cartesian coordinates (time, x, y, z)

    Args:
        data(numpy array) = containing the spherical coordinates in format of (time, azimuth, elevation, r)

    Returns:
        result(numpy array) = containing the cartesian coordinates in format of (time, x, y, z)
    '''

    for i in range(0, len(data)):
        elevation = data[i, 1]
        azimuth = data[i, 2]
        r = data[i, 3]

        x = r * cos(elevation) * cos(azimuth)
        y = r * cos(elevation) * sin(azimuth)
        z = r * sin(elevation)

        result = data
        result[i, 1] = x
        result[i, 2] = y
        result[i, 3] = z

    return result

if __name__ == "__main__":

    data = read_data.load_data("orbit.csv")
    new_data = cart_to_spher(data)
    same_data = spher_to_cart(new_data)
    print(same_data == data)

