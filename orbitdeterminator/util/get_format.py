'''
Checks and return the format (Cartesian, U.K., I.O.D. or R.D.E) used in the file given as argument
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import datetime

def checkdate(date):
	'''
	Checks if given string is a valid date in YYYYMMDD fromat

	Args:
		date (string): A string of length '8' representing YYYYMMDD

	Returns:
		Boolean :-
			True: If given date is a valid date
			False: If date is invalid
	'''
	try:
		datetime.datetime.strptime(date, '%Y%m%d')
		return True
	except ValueError:
		return False

def get_format(data_file):
	'''
	Detects the format of given file

	Args:
		date_file (string): Location of the file inside the orbitdeterminator/obs_data folder

	Returns:
		String :-
			'Cartesian': If observation is in Cartesian coordinate system
			'RDE': If observation is in R.D.E. format
			'IOD': If observation is in I.O.D. format
			'UK': If observation is in U.K. format
			'Undefined format': If observation format is not among R.D.E., I.O.D. or U.K. format
			Error message if data file extension is neither .txt nor .csv
	'''
	Error = "Please use data in '.txt' or '.csv' file formats"
	if data_file[-4:] == ".txt":
		lines = [line.rstrip('\n') for line in open(data_file)]
	elif data_file[-4:] == ".csv":
		lines = np.genfromtxt(data_file, delimiter='\t')
	else:
		return Error
	length = [(len(line)) for line in lines]
	if length[0] == 4:
		return 'Cartesian'
	elif length[0] == 20:
		return 'RDE'
	elif length[0] <= 80:
		if (checkdate(lines[0][23:31])):
			return 'IOD'
		elif (checkdate('20'+lines[0][11:17])):
			return 'UK'
		else:
			return 'Undefined format'
	else:
		return "Undefined format"

if __name__ == "__main__":
	
	data_file = "../obs_data/SourceTXT/uk_sample.txt"
	print(get_format(data_file))