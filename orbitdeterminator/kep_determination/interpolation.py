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
from scipy.interpolate import CubicSpline

from util import state_kep
from util import read_data


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
	coordinates = list([orbit_data[:,1:2], orbit_data[:,2:3], orbit_data[:,3:4]])
	splines = list(map(lambda a:CubicSpline(time.ravel(),a.ravel()), coordinates))

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

	return np.array(velocity)

def main():
	data_points = read_data.load_data("../filtered.csv")

	splines = cubic_spline(data_points)

	points = data_points[:,1:4].tolist()

	keplerians = [] # Keplerian elements for each of the points
	
	for point in points:
		keplerians.append(state_kep.state_kep(point, compute_velocity(splines, point)))

	print(np.array(keplerians).mean(axis=0))



if __name__ == "__main__":

	main()

