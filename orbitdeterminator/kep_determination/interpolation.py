'''
Author: Nilesh Chaturvedi
Date Created:4th July, 2017

Description: Interpoaltion using splines for calculating velocity at a point 
and hence the orbital elements.
'''
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import scipy
import numpy as np

from orbitdeterminator.util import state_kep
from orbitdeterminator.util import read_data


def compute_keplerians(position, velocity):
	''' Compute orbital elements from the state vectors(position, velocity).

	Args:
		position (numpy array): position vector of a point
		velocity (numpy array): velocity vector at that point

	Returns:
		keplerians (numpy array): orbital elements based on the state vector.
	'''
	pass

def quadratic_spline(data_points):
	''' Compute spline of degree 2 between 2 intermediate points of input
		
		Args:
			data_points (numpy array): array of orbit data points.

		Returns:
			spline_array (numpy array): array of quadratic splines of orbit data points 
	'''
	pass

def cubic_spline(orbit_data):
	''' Compute component wise cubic spline of points of input data
		
		Args:
			orbit_data (numpy array): array of orbit data points of the 
				format [time, x, y, z]

		Returns:
			splines (list): component wise cubic splines of orbit data points 
				of the format [spline_x, spline_y, spline_z]
	'''
	time = orbit_data[:,:1]
	co-ordinates = list([orbit_data[:,1:2], orbit_data[:,2:3], orbit_data[:,3:4]])
	splines = list(map(lambda a:CubicSpline(time.ravel(),a.ravel()), co-ordinates))

	return splines

def compute_velocity(spline, point):
	''' Calculate the velocity at a point given the spline.

	Calculate the deraivative of spline at the point(on the points the 
	given spline corresponds to). This gives the velocity at that point.

	Args:
		spline (list): component wise cubic splines of orbit data points 
			of the format [spline_x, spline_y, spline_z].
		point (numpy array): point at which velocity is to be calculated.

	Returns:
		velocity (numpy array): velocity vector at the given point
	'''
	velocity = list(map(lambda s, x:s(x, 1), spline, point))

	return numpy.array(velocity)

if __name__ == "__main__":

	pass