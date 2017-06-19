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
	Returns a 2D numpy array of the orbit
	Each point of the orbit is of the 
	format : [timestamp, x-coordinate, y-coordinate, z-coordinate]
	'''
	orbit_file = list(csv.reader(open(filename, "r"), delimiter = ","))

	orbit = []
	for point in orbit_file:
		point_tuple = np.array(point, dtype = np.float)
		orbit.append(point_tuple)

	return np.array(orbit)

if __name__ == "__main__":
	'''Returns a dictionary of the format {filename: orbit}'''
	
	parsed_orbits = {}
	files = os.listdir(os.getcwd() + '/' + sys.argv[1])

	for file in files:
		if file.endswith('.csv'):
			orbit = load_data(os.getcwd() + '/' + sys.argv[1] + '/' + str(file))
			parsed_orbits[file] = orbit

	pickle.dump(parsed_orbits, open("orbits.p", "wb"))

	print("Object file created with the name 'orbits.p'!!")