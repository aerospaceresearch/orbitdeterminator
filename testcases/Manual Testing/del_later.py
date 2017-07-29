import csv
import os
import pickle
import numpy
path_meta = os.getcwd() + "/meta1/"

path_tracks = os.getcwd() + "/track1/"

files_true = os.listdir(path_meta)
files_exp = os.listdir(path_tracks)
print(files_true)
true_vals = {}

###################################################################
################ Get dictionary of true values ####################
for i in files_true:
	print(i)
	a = list(csv.reader(open(path_meta + i, "r"), delimiter = "\t"))[1]
	true_vals[a[0][16:30]] = numpy.array(a[4:10], dtype = numpy.float32)

###################################################################
print(true_vals)
pickle.dump(true_vals, open("true_kep_dict.p", 'wb'))