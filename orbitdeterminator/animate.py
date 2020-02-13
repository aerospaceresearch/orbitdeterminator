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


fig = plt.figure()
ax = plt.axes(projection="3d")


x=[]
y=[]
z=[]
i=0

data='orbit.csv'
def read(data_file):
	global data
	data = read_data.load_data(data_file)
	data[:, 1:4] = data[:, 1:4] / 1000
	data_after_filter = wiener.wiener_new(data,3)
	return data_after_filter

    
data_after_filter=read(data)
def animate(i):
	
       
    x.append(data_after_filter[i*100][1])
    y.append(data_after_filter[i*100][2])
    z.append(data_after_filter[i*100][3])
    plt.plot(x,y,z,'b--')
    
    print(i,data_after_filter[i][1],data_after_filter[i][2],data_after_filter[i][3])
    i=i+100
    

        
def main(data_file):
	print('Do you want to animate orbit Y/N')
	choice=input()
	if(choice in 'Yy'):
		
		global data_after_filter
		data_after_filter=read(data_file)
		ani=FuncAnimation(plt.gcf(),animate,interval=100)
		plt.show()
	else:
		return None


