
from vpython import *
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation
import random
import math
from mpl_toolkits import mplot3d
from util import (read_data, kep_state, rkf78, golay_window)
from filters import (sav_golay, triple_moving_average,wiener)
from kep_determination import (lamberts_kalman, interpolation, ellipse_fit, gibbsMethod)

'''
This is a module to animate orbit of a satellite around a planet of given radius r and data file.
'''






#data='orbit.csv'
def read(data_file):
	#global data
	data = read_data.load_data(data_file)
	data[:, 1:4] = data[:, 1:4] / 1000
	data_after_filter = wiener.wiener_new(data,3)
	return data_after_filter

def animate(data_file,r):
	print("Do you want to animate orbit Y/N")
	choice=input()
	if(choice in 'Yy'):
		i=0    
		data_after_filter=read(data_file)
		c=data_after_filter.shape[0]
		Earth=sphere(pos=vector(0,0,0), radius=r, color=color.red)
		satellite=sphere(pos=vector(data_after_filter[i][1],data_after_filter[i][2],data_after_filter[i][3]), radius=100, color=color.blue,make_trail=True, trail_type="points",
              interval=3, retain=30,trail_color=color.green,trail_radius=100)
		k=100

		while True:
		    rate(500)
		    satellite.pos.x=data_after_filter[i%c][1]
		    satellite.pos.y=data_after_filter[i%c][2]
		    satellite.pos.z=data_after_filter[i%c][3]
		    i+=1
		    #print(satellite.pos.x,satellite.pos.y,satellite.pos.z)
	else:
		return 0
	    


