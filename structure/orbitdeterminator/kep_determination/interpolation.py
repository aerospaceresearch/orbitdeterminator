'''
Author: Nilesh Chaturvedi
Date Created:4th July, 2017

Description: Interpoaltion using splines for calculating velocity at a point 
and hence the orbital elements.
'''

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

def cubic_spline(data_points):
	''' Compute polynomial spline of degree 3 between 2 intermediate points of input
		
		Args:
			data_points (numpy array): array of orbit data points.

		Returns:
			spline_array (numpy array): array of cubic splines of orbit data points 
	'''
	pass

def compute_velocity(spline, point):
	''' Calculate the velocity at a point given the spline.

	Calculate the deraivative of spline at the point(on the points the 
	given spline corresponds to). This gives the velocity at that point.

	Args:
		spline (numpy array): spline function of two points
		point (numpy array): one of the two points of the spline on which derivative
			is to be calculated.

	Returns:
		velocity (numpy array): velocity vector at the given point
	'''

if __name__ == "__main__":

	pass