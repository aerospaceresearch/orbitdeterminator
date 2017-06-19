'''
Created by Alexandros Kazantzidis
Date 25/05/17
'''

import numpy as np
from math import *

def Mtov (M,e):

	'''
	Computes true anomaly v from a given mean anomaly M and eccentricity e by using Newton-Raphson method

	Args:
		M(float) = mean anomaly (degrees)
		e(float) = eccentricity (number)
	
	Returns:
		v(float) = true anomaly (degrees)'''

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
    Uses the keplerian elements to compute the position and velocity vector

    Args:
		kep(numpy array) = a 1x6 matrix which contains the following variables
			kep(0)=inclination (degrees)
			kep(1)=right ascension of the ascending node (degrees)
			kep(2)=eccentricity (number)
			kep(3)=argument of perigee (degrees)
			kep(4)=mean anomaly (degrees)
			kep(5)=mean motion (revs per day)

	Returns:
		r(numpy array) = 1x6 matrix which contains the position and velocity vector
		r(0),r(1),r(2) = position vector (rx,ry,rz) m
		r(3),r(4),r(5) = velocity vector (vx,vy,vz) m/s
	'''
	
	
	
	r = np.zeros((6, 1))
	mu = 398600.4405
	
	# unload orbital elements array
	

	sma = kep[0,0]
	ecc = kep[1,0] # eccentricity
	inc = kep[2,0] # inclination
	argper = kep[3,0] # argument of perigee
	raan = kep[4,0]# right ascension of the ascending node
	tanom = kep[5,0] # we use mean anomaly(kep(5)) and the function Mtov to compute true anomaly (tanom)
	
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





