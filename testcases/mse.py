'''
Author: Nilesh Chaturvedi
Date created: 2nd June, 2017

Calculates mean error between true value and smooth value
arg[1]: true data, 
arg[2]: filtered data
'''
import os
import sys
import numpy
import read_data as rd

def error(true, filtered):
	'''
	Calculated the mean square error between the arrays of the two numpy matrices

	Args:
	true: reference matrix
	filtered: objective matrix

	Returns:
	percentage error between the two matrices 
	'''
	error_x = numpy.absolute(true[:,1] - filtered[:,1])
	error_y = numpy.absolute(true[:,2] - filtered[:,2])
	error_z = numpy.absolute(true[:,3] - filtered[:,3])

	return numpy.mean(numpy.dstack((error_x, error_y, error_z)))

if __name__ == "__main__":

	true = os.getcwd() + '/' + sys.argv[1]
	target = os.getcwd() + '/' + sys.argv[2]

	true_val = rd.load_data(true)
	filtered_val = rd.load_data(target)

	print(error(true_val, filtered_val))