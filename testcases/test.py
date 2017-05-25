import os
import sys
import csv
import pickle
import numpy as np


parsed_orbits = {}
	
for file in os.listdir(os.getcwd() + '/tests'):
	print(file)
	if file.endswith('.csv'):
		print(file)
		orbit_file = list(csv.reader(open(os.getcwd() + '/tests/' + file, "r"), delimiter = ","))
		orbit = []
		
		for point in orbit_file:
			point_tuple = np.array(point, dtype = np.float)
			orbit.append(point_tuple)
		
		parsed_orbits[file] = orbit

print(parsed_orbits)
