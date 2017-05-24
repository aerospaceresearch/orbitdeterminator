'''
Author : Nilesh Chaturvedi
Date Created : 21/05/2017
'''

import os
import sys
import csv
import numpy as np

def load_data(filename):
	'''
	Returns a list of numpy arrays with each array of the format 
	[timestamp, x-coordinate, y-coordinate, z-coordinate]
	'''

	data_list = list(csv.reader(open(filename, "r"), delimiter = ","))
	orbit = []
	for i in data_list:
		data_tup = np.array(i, dtype = np.float)
		orbit.append(data_tup)

	return orbit

if __name__ == "__main__":

	orbit = load_data('orbit0perfect.csv')

	print("[Time \tX\tY\tZ ]")

	for point in orbit:
		print(point)
