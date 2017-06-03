'''
Created by Alexandros Kazantzidis
Date 10/02/17
'''


from math import *
from decimal import *
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)

np.set_printoptions(precision=16)


def hello1(x):

	x_parag=(14.0*sin(x/600.0))
	return x_parag

def ypol_a(y):
	
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
	
	mu=398600.4405;
	y_parag = np.zeros((6,1))
	agrav = np.zeros((3,1))
	
	
	r2 = y[0,0]*y[0,0] + y[1,0]*y[1,0] + y[2,0]*y[2,0]
	r1 = sqrt(r2)
	r3 = r2*r1
	
	for i in range(0,3):
		agrav[i,0] = agrav[i,0] -(mu * y[i,0] / r3)
		
	
	y_parag[0,0]=y[3,0]
	y_parag[1,0]=y[4,0]
	y_parag[2,0]=y[5,0]
	y_parag[3,0]=agrav[0,0]
	y_parag[4,0]=agrav[1,0]
	y_parag[5,0]=agrav[2,0]
	return y_parag
		
	


def rkf78 (neq,ti,tf,h,tetol,x):

	# solve first order system of differential equations

	# Runge-Kutta-Fehlberg 7[8] method

	# input

	#  deq   = name of function which defines the
	#          system of differential equations
	#  neq   = number of differential equations
	#  ti    = initial simulation time
	#  tf    = final simulation time
	#  h     = initial guess for integration step size
	#  tetol = truncation error tolerance [non-dimensional]
	#  x     = integration vector at time = ti

	# output

	#  xout  = integration vector at time = tf




	




	# allocate arrays



   # define integration coefficients

	ch = np.zeros((13,1))
	alph = np.zeros((13,1))
	beta = np.zeros((13, 12))
	
	ch[5,0] = 34.0 / 105
	ch[6,0] = 9.0 / 35
	ch[7,0] = ch[6,0]
	ch[8,0] = 9.0 / 280
	ch[9,0] = ch[8,0]
	ch[11,0] = 41.0 / 840
	ch[12,0] = ch[11,0]
 
	alph[1,0] = 2.0 / 27
	alph[2,0] = 1.0 / 9
	alph[3,0] = 1.0 / 6
	alph[4,0] = 5.0 / 12
	alph[5,0] = 0.5
	alph[6,0] = 5.0 / 6
	alph[7,0] = 1.0 / 6
	alph[8,0] = 2.0 / 3
	alph[9,0] = 1.0 / 3
	alph[10,0] = 1
	alph[12,0] = 1
	

	beta[1,0] = 2.0 / 27
	beta[2,0]  = 1.0 / 36
	beta[3,0] = 1.0 / 24
	beta[4,0] = 5.0 / 12
	beta[5,0]   = 0.05
	beta[6,0] = -25.0 / 108
	beta[7,0] = 31.0 / 300
	beta[8,0] = 2.0
	beta[9,0] = -91.0 / 108
	beta[10,0] = 2383.0 / 4100
	beta[11,0]  = 3.0 / 205
	beta[12,0] = -1777.0 / 4100
	beta[2,1] = 1.0 / 12
	beta[3,2] = 1.0 / 8
	beta[4,2] = -25.0 / 16
	beta[4,3] = -beta[4,2]
	beta[5,3] = 0.25
	beta[6,3] = 125.0 / 108
	beta[8,3] = -53.0 / 6
	beta[9,3] = 23.0 / 108
	beta[10,3] = -341.0 / 164
	beta[12,3] = beta[10,3]
	beta[5,4] = 0.2
	beta[6,4] = -65.0 / 27
	beta[7,4] = 61.0 / 225
	beta[8,4] = 704.0 / 45
	beta[9,4] = -976.0 / 135
	beta[10,4] = 4496.0 / 1025
	beta[12,4] = beta[10,4]
	beta[6,5] = 125.0 / 54
	beta[7,5] = -2.0 / 9
	beta[8,5]  = -107.0 / 9
	beta[9,5] = 311.0 / 54
	beta[10,5] = -301.0 / 82
	beta[11,5] = -6.0 / 41
	beta[12,5] = -289.0 / 82
	beta[7,6] = 13.0 / 900
	beta[8,6] = 67.0 / 90
	beta[9,6] = -19.0 / 60
	beta[10,6] = 2133.0 / 4100
	beta[11,6] = -3.0 / 205
	beta[12,6] = 2193.0 / 4100
	beta[8,7] = 3.0
	beta[9,7] = 17.0 / 6
	beta[10,7] = 45.0 / 82
	beta[11,7] = -3.0 / 41
	beta[12,7] = 51.0 / 82
	beta[9,8] = -1.0 / 12
	beta[10,8] = 45.0 / 164
	beta[11,8]  = 3.0 / 41
	beta[12,8] = 33.0 / 164
	beta[10,9] = 18.0 / 41
	beta[11,9] = 6.0 / 41
	beta[12,9] = 12.0 / 41
	beta[12,11] = 1.0

	
	f = np.zeros((neq,13));

	xdot = np.zeros((neq,1));

	xwrk = np.zeros((neq, 1));

	# compute integration "direction"
	dt=h
	
	while True:

	# load "working" time and integration vector

		twrk = ti  
		xwrk[:,0] = x[:,0]

	# check for last dt

		if abs(dt) > abs(tf - ti):
			dt = tf - ti
	
	# check for end of integration period
		
		if abs(ti - tf) < 0.00000001:
			xout = x
			return xout

			
		xdot = ypol_a(x)
		xdot_tra=np.transpose(xdot)
		f[:, 0] = xdot_tra
		
		for k in range(1,13):
			
      
			for i in range(0,neq):
				
				x[i,0] = xwrk[i,0] + dt * sum(beta[k, 0:k] * f[i, 0:k])
				ti = twrk + alph[k,0] * dt;
				xdot = ypol_a(x);
				xdot_tra=np.transpose(xdot)
				f[:,k] = xdot_tra;
		
		xerr = tetol;
		for i in range(0,neq):
			f_tra=np.transpose(f)
			x[i,0] = xwrk[i,0] + dt * sum(ch[:,0] * f_tra[:,i])
			
			# truncation error calculations

			ter = abs((f[i, 0] + f[i, 10] - f[i, 11] - f[i, 12]) * ch[11,0] * dt)
			tol = abs(x[i,0]) * tetol + tetol
			tconst = ter / tol
		
		if tconst > xerr: xerr = tconst
##
		# compute new step size

		dt = 0.8 * dt * (1.0 / xerr) ** (1.0 / 8)
		
		if (xerr > 1):
		# reject current step
			ti = twrk
			x = xwrk
		
		
		
	
if __name__ == "__main__":
	neq=6
	ti=1.0
	tf=100.0
	h=0.1
	tetol=1e-04
	x= np.array([[1.51303397e+03],[-2.48429276e+03],[6.46549360e+03],[2.99258730e+00],[-6.15860507e+00],[-3.06500279e+00]])

	xout = rkf78(neq,ti,tf,h,tetol,x)
	print(sqrt(xout[0]**2+xout[1]**2+xout[2]**2))
	print(xout)

