'''
Created by Alexandros Kazantzidis
Date 25/05/17
'''

import numpy as np
from math import *
import pandas as pd

def Mtov (M,e):

	'''
	Computes true anomaly v from a given mean anomaly M and eccentricity e by using Newton-Raphson method 

	input
	M = mean anomaly (degrees)
	e = eccentricity (number)
	
	output
	
	v = true anomaly (degrees)'''

	i=1
	Eo=100
	while True:
		E=Eo-((Eo-e*sin(Eo)-M)/(1-e*cos(Eo)))
		Eo=E
		i=i+1
		if i==100: break
	
	v_pre=(cos(E)-e)/(1-e*cos(E))
	v= acos(v_pre)
	v = degrees(v)
	return v
	
	


	
def Kep_state (kep):
	'''
    this function uses the keplerian elements to compute the position and velocity vector 

    input

	kep is a 1x6 matrix which contains the following variables
	kep(0)=inclination (degrees)
	kep(1)=right ascension of the ascending node (degrees)
	kep(2)=eccentricity (number)
	kep(3)=argument of perigee (degrees)
	kep(4)=mean anomaly (degrees)
	kep(5)=mean motion (revs per day)

	output

	r = 1x6 matrix which contains the position and velocity vector
	r(0),r(1),r(2) = position vector (rx,ry,rz) m
	r(3),r(4),r(5) = velocity vector (vx,vy,vz) m/s
	'''
	
	
	
	r = np.zeros((6,1))
	mu = 398600.4405
	
	# unload orbital elements array
	
	sma_pre= (398600.4405*(86400**2))/((kep[5,0]**2)*4*(pi**2));
	sma = sma_pre**(1.0/3.0) # sma is semi major axis, we use mean motion (kep(6)) to compute this
	ecc = kep[2,0] # eccentricity
	inc = kep[0,0] # inclination
	argper = kep[3,0] # argument of perigee
	raan = kep[1,0];# right ascension of the ascending node
	tanom = Mtov(kep[4,0],ecc) # we use mean anomaly(kep(5)) and the function Mtov to compute true anomaly (tanom)
	
	tanom=radians(tanom)
	slr = sma * (1 - ecc * ecc)
	rm = slr / (1 + ecc * cos(tanom))
	tanom=degrees(tanom)
	
	

	arglat = argper + tanom; # argument of latitude
	
	arglat=radians(arglat)
	sarglat = sin(arglat)
	carglat = cos(arglat)
	arglat=degrees(arglat)
	
	
	argper=radians(argper)
	c4 = sqrt(mu / slr)
	c5 = ecc * cos(argper) + carglat
	c6 = ecc * sin(argper) + sarglat
	argper=degrees(argper)
	
	
	inc=radians(inc)
	sinc = sin(inc)
	cinc = cos(inc)
	inc=degrees(inc)
	
	raan=radians(raan)
	sraan = sin(raan)
	craan = cos(raan)
	raan=degrees(raan)
	
	
	
	
	
		# position vector

	r[0,0] = rm * (craan * carglat - sraan * cinc * sarglat)
	r[1,0] = rm * (sraan * carglat + cinc * sarglat * craan)
	r[2,0] = rm * sinc * sarglat

		# velocity vector

	r[3,0] = -c4 * (craan * c6 + sraan * cinc * c5)
	r[4,0] = -c4 * (sraan * c6 - craan * cinc * c5)
	r[5,0] = c4 * c5 * sinc
	
	return r

if __name__ == "__main__":


	kep = np.array([[98.5517],[271.9207],[0.0002336],[137.9790],[222.1574],[14.34543485]])

	r = Kep_state (kep)
	df = pd.DataFrame (r)
	print (df)



