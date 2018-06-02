"""Computes the least-squares optimal Keplerian elements for a sequence of
   cartesian position observations.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# convention:
# a: semi-major axis
# e: eccentricity
# eps: mean longitude at epoch
# Euler angles:
# I: inclination
# Omega: longitude of ascending node
# omega: argument of pericenter

#rotation about the z-axis about an angle `ang`
def rotz(ang):
    cos_ang = math.cos(ang)
    sin_ang = math.sin(ang)
    return np.array(((cos_ang,-sin_ang,0.0), (sin_ang, cos_ang,0.0), (0.0,0.0,1.0)))

#rotation about the x-axis about an angle `ang`
def rotx(ang):
    cos_ang = math.cos(ang)
    sin_ang = math.sin(ang)
    return np.array(((1.0,0.0,0.0), (0.0,cos_ang,-sin_ang), (0.0,sin_ang,cos_ang)))

#rotation from the orbital plane to the inertial frame
#it is composed of the following rotations, in that order:
#1) rotation about the z axis about an angle `omega` (argument of pericenter)
#2) rotation about the x axis about an angle `I` (inclination)
#3) rotation about the z axis about an angle `Omega` (longitude of ascending node)
def op2if(omega,I,Omega):
    P2_mul_P3 = np.matmul(rotx(I),rotz(omega))
    return np.matmul(rotz(Omega),P2_mul_P3)

# # TODO:
# # write function to compute true anomaly as a function of time-of-fly
# # write function to compute range as a function of orbital elements
# # write function which takes observed values and computes the difference wrt expected-to-be-observed values as a function of unknown orbital elements (to be fitted)
# # compute Q as a function of unknown orbital elements (to be fitted)
# # optimize Q -> return fitted orbital elements (requires an ansatz: take input from minimalistic Gibb's?)

# NOTES:
# matrix multiplication of numpy's 2-D arrays is done through `np.matmul`

omega = math.radians(31.124)
I = math.radians(75.0)
Omega = math.radians(60.0)

# rotation matrix from orbital plane to inertial frame
# two ways to compute it; result should be the same
P_1 = rotz(omega) #rotation about z axis by an angle `omega`
P_2 = rotx(I) #rotation about x axis by an angle `I`
P_3 = rotz(Omega) #rotation about z axis by an angle `Omega`

Rot1 = np.matmul(P_3,np.matmul(P_2,P_1))
Rot2 = op2if(omega,I,Omega)

v = np.array((3.0,-2.0,1.0))

print(I)
print(omega)
print(Omega)

print(Rot1)

print(np.matmul(Rot1,v))

print(Rot2)