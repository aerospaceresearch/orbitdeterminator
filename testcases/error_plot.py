'''
Author : Nilesh Chaturvedi
Date Created : 12th June, 2017

Shows the error graph for varying filter size in triple_moving_average.py
'''
import os
import mse
import numpy
import matplotlib.pyplot as plt

import read_data as rd
import tripple_moving_average as tma

signal = rd.load_data(os.getcwd() + '/orbit0jittery.csv')
perfect = rd.load_data(os.getcwd() + '/orbit0perfect.csv')

errors = []
for i in range(2, 100):
	filtered = tma.generate_filtered_data(signal, i)
	error = mse.error(perfect, filtered)
	errors.append(error)

# Plot Error vs window size graph
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(numpy.arange(2,100), errors, 'b', label='filtered')
ax.legend(['window size vs error'])
plt.show()

