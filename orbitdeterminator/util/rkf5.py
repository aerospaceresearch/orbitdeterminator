"""Old script documentation and code is kinda bad"""

from math import *
from decimal import *
import numpy as np


def ypol_a(t,y):

	# function which computes the 1x6 vector y_parag (contains velocity and
	# acceleration values) by using the state vector y
	# keplerian motion to initialize the acceleration vector

	# input

	# y = state vector (y(1),y(2),y(3) = position vector and y(4),y(5),y(6) = velocity vector)
	# y(1),y(2),y(3) m and y(4),y(5),y(6) m/s

	# output

	# y_parag = y' vector which contains the velocity and acceleration values
	# y_parag(1,2,3) = velocity vector and y_parag(4,5,6) = acceleration vector
	# y_parag(1,2,3) = m/s and y_parag(4,5,6) = m/s^2

	mu_Earth=398600.4405
	y_parag = np.zeros((6,1))
	agrav = np.zeros((3,1))


	r2 = y[0,0]*y[0,0] + y[1,0]*y[1,0] + y[2,0]*y[2,0]
	r1 = sqrt(r2)
	r3 = r2*r1

	for i in range(0,3):
		agrav[i,0] = agrav[i,0] -(mu_Earth * y[i,0] / r3)


	y_parag[0,0]=y[3,0]
	y_parag[1,0]=y[4,0]
	y_parag[2,0]=y[5,0]
	y_parag[3,0]=agrav[0,0]
	y_parag[4,0]=agrav[1,0]
	y_parag[5,0]=agrav[2,0]
	return y_parag


def rkf5 (ti,tf,h,x,neq=6):

	# RKF45 method
	#input

	#  deq   = name of function which defines the
	#          system of differential equations
	#  neq   = number of differential equations
	#  ti    = initial simulation time
	#  tf    = final simulation time
	#  h     = initial guess for integration step size
	#  emin = minimum error we require from the solution
	#  emax = max error we require from the solution
	#  x     = integration vector at time = ti

	# output

	#  x1  = integration vector at time = tf

	##
	yoRKF5 = np.zeros((6,1))
	K1neo = np.zeros((6,1))
	K2neo =  np.zeros((6,1))
	K3neo =  np.zeros((6,1))
	K4neo =  np.zeros((6,1))
	K5neo =  np.zeros((6,1))
	K6neo =  np.zeros((6,1))
	y1RKF5 =  np.zeros((6,1))
	yoRKF5[:]=x[:]
	ta=ti
	epoch_solution = []
	keep = []
		##

	while True:
		dt=tf-ti
		tt=ti+h


		if (abs(h)>abs(dt)):
			h=dt

		##
		for j in range(0,1):


			#parametroi K gia RKF5
			K1neo = h * ypol_a(ti,yoRKF5)

			t2=ti+(1.0/4)*h
			yo2neo=yoRKF5+(1/4)*K1neo
			K2neo = h * ypol_a(t2,yo2neo)

			t3=ti+(3.0/8)*h
			yo3neo=yoRKF5+(3.0/32)*K1neo+(9.0/32)*K2neo
			K3neo=h*ypol_a(t3,yo3neo)

			t4=ti+(12.0/13)*h
			yo4neo=yoRKF5+(1932.0/2197)*K1neo-(7200.0/2197)*K2neo+(7296.0/2197)*K3neo
			K4neo=h*ypol_a(t4,yo4neo)

			t5=ti+h
			yo5neo=yoRKF5+(439.0/216)*K1neo-8*K2neo+(3680.0/513)*K3neo-(845.0/4104)*K4neo
			K5neo=h*ypol_a(t5,yo5neo)

			t6=ti+(1.0/2)*h
			yo6neo=yoRKF5-(8.0/27)*K1neo+2*K2neo-(3544.0/2565)*K3neo+(1859.0/4104)*K4neo-(11.0/40)*K5neo
			K6neo=h*ypol_a(t6,yo6neo)
	##

			y1RKF5=yoRKF5+(16.0/135)*K1neo+(6656.0/12825)*K3neo+(28561.0/56430)*K4neo-(9.0/50)*K5neo+(2.0/55)*K6neo

			yoRKF5[:]=y1RKF5[:]


		epoch_solution.append(y1RKF5)
		ti=ti+h

		if (abs(ti - tf) < 0.00000001):

			xrkf5=y1RKF5
			final = np.zeros((len(epoch_solution), 6))
			for i in range(0, len(epoch_solution)):
				final[i, :] = np.ravel(epoch_solution[i])

			return xrkf5, final


if __name__ == "__main__":
	# Starting time
	ti = 1.0
	# Final time
	tf = 1000.0
	# Step of the algo, it will give one state vector for every 10sec from ti to tf
	h = 10.0
	# Initial State vector
	x = np.array([[1.51303397e+03],[-2.48429276e+03],[6.46549360e+03],[2.99258730e+00],[-6.15860507e+00],[-3.06500279e+00]])

	xrkf5, every = rkf5(ti,tf,h,x)
	# Every is the state vector from ti to tf for each step
	print(every)
