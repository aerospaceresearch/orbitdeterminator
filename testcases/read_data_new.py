'''
Author : Nilesh Chaturvedi
Date Created : 21/05/2017
'''

import os
import sys
import csv
import pickle
import numpy as np

def load_data(folder_name):
	'''
	Returns a dictionary with the following format :
	{
		'filename1' : orbit1
		'filename2' : orbit2
		.
		.
		.
	
		}

	Each point of the orbit is of the 
	format : [timestamp, x-coordinate, y-coordinate, z-coordinate]
	'''

	parsed_orbits = {}
	
	for file in os.listdir(os.getcwd() + '/' + folder_name):
		if file.endswith('.csv'):

			orbit_file = list(csv.reader(open(os.getcwd() + '/' + folder_name + '/' + str(file), "r"), delimiter = ","))
			
			orbit = []
			
			for point in orbit_file:
				point_tuple = np.array(point, dtype = np.float)
				
				orbit.append(point_tuple)
			parsed_orbits[file] = orbit

	return parsed_orbits

if __name__ == "__main__":

	orbits = load_data(sys.argv[1])

	pickle.dump(orbits, open("orbits.p", "wb"))

	print("Object file created with the name 'orbits.p'!!")


