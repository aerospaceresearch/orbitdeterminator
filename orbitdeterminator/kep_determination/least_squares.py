"""Computes the least-squares optimal Keplerian elements for a sequence of
   cartesian position observations.
"""

import math
#import numpy as np
#Thinly-wrapped numpy
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from autograd import jacobian
from autograd import elementwise_grad as egrad

# convention:
# a: semi-major axis
# e: eccentricity
# eps: mean longitude at epoch
# taup: time of pericenter passage
# Euler angles:
# I: inclination
# Omega: longitude of ascending node
# omega: argument of pericenter

#rotation about the z-axis about an angle `ang`
def rotz(ang):
    cos_ang = np.cos(ang)
    sin_ang = np.sin(ang)
    return np.array(((cos_ang,-sin_ang,0.0), (sin_ang, cos_ang,0.0), (0.0,0.0,1.0)))

#rotation about the x-axis about an angle `ang`
def rotx(ang):
    cos_ang = np.cos(ang)
    sin_ang = np.sin(ang)
    return np.array(((1.0,0.0,0.0), (0.0,cos_ang,-sin_ang), (0.0,sin_ang,cos_ang)))

#rotation from the orbital plane to the inertial frame
#it is composed of the following rotations, in that order:
#1) rotation about the z axis about an angle `omega` (argument of pericenter)
#2) rotation about the x axis about an angle `I` (inclination)
#3) rotation about the z axis about an angle `Omega` (longitude of ascending node)
def orbplane2frame_(omega,I,Omega):
    P2_mul_P3 = np.matmul(rotx(I),rotz(omega))
    return np.matmul(rotz(Omega),P2_mul_P3)

def orbplane2frame(x):
    return orbplane2frame_(x[0],x[1],x[2])

# get Keplerian range
def kep_r_(a, e, f):
    return a*(1.0-e**2)/(1.0+e*np.cos(f))

def kep_r(x):
    return kep_r_(x[0],x[1],x[2])

# get cartesian positions wrt orbital plane
def xyz_orbplane_(a, e, f):
    r = kep_r_(a, e, f)
    return np.array((r*np.cos(f),r*np.sin(f),0.0))

def xyz_orbplane(x):
    return xyz_orbplane_(x[0],x[1],x[2])

# get cartesian positions wrt inertial frame from orbital elements
def xyz_frame_(a,e,f,omega,I,Omega):
    return np.matmul( orbplane2frame_(omega,I,Omega) , xyz_orbplane_(a, e, f) )

def xyz_frame(x):
    return np.matmul( orbplane2frame(x[3:6]) , xyz_orbplane(x[0:3]) )

# get mean motion from mass parameter (mu) and semimajor axis (a)
def meanmotion(mu,a):
    return np.sqrt(mu/(a**3))

# get mean anomaly from mean motion (n), time (t) and time of pericenter passage (taup)
def meananomaly(n, t, taup):
    return n*(t-taup)

# compute eccentric anomaly (E) from eccentricity (e) and mean anomaly (M)
def eccentricanomaly(e,M):
    M0 = np.mod(M,2*np.pi)
    E0 = M0 + np.sign(np.sin(M0))*0.85*e #Murray-Dermotts' initial estimate
    # successive approximations via Newtons' method
    for i in range(0,4):
        #TODO: implement modified Newton's method for Kepler's equation (Murray-Dermott)
        Eans = E0 - (E0-e*np.sin(E0)-M0)/(1.0-e*np.cos(E0))
        E0 = Eans
    return E0

#compute true anomaly (f) from eccentricity (e) and eccentric anomaly (E)
def trueanomaly(e,E):
    Enew = np.mod(E,2*np.pi)
    return 2*np.arctan(  np.sqrt((1.0+e)/(1.0-e))*np.tan(Enew/2)  )



# # TODO:
# # write function to compute range as a function of orbital elements: DONE
# # write function to compute true anomaly as a function of time-of-fly
# # the following transformation is needed: from time t, to mean anomaly M,
# # to eccentric anomaly E, to true anomaly f
# # t -> M=n*(t-taup) -> M=E-e*sin(E) (invert) ->
# # -> f = 2*atan(  sqrt((1+e)/(1-e))*tan(E/2)  )
# # write function which takes observed values and computes the difference wrt expected-to-be-observed values as a function of unknown orbital elements (to be fitted)
# # compute Q as a function of unknown orbital elements (to be fitted)
# # optimize Q -> return fitted orbital elements (requires an ansatz: take input from minimalistic Gibb's?)

# NOTES:
# matrix multiplication of numpy's 2-D arrays is done through `np.matmul`

omega = np.radians(31.124)
I = np.radians(75.0)
Omega = np.radians(60.0)

# rotation matrix from orbital plane to inertial frame
# two ways to compute it; result should be the same
P_1 = rotz(omega) #rotation about z axis by an angle `omega`
P_2 = rotx(I) #rotation about x axis by an angle `I`
P_3 = rotz(Omega) #rotation about z axis by an angle `Omega`

Rot1 = np.matmul(P_3,np.matmul(P_2,P_1))
Rot2 = orbplane2frame_(omega,I,Omega)

v = np.array((3.0,-2.0,1.0))

print(I)
print(omega)
print(Omega)

print(Rot1)

print(np.matmul(Rot1,v))

print(Rot2)

# #import autograd.numpy as np  # Thinly-wrapped numpy
# #from autograd import grad    # The only autograd function you may ever need
# def tanh(x):                 # Define a function
#     y = np.exp(-2.0 * x)
#     return (1.0 - y) / (1.0 + y)

# grad_tanh = grad(tanh)       # Obtain its gradient function
# print(grad_tanh(1.0))

# def mycos(x):
#     return np.cos(x)

# drotz = grad(mycos)

# print(drotz(0.1))

drotz = jacobian(orbplane2frame_)
drotz2 = jacobian(orbplane2frame)

print('op2f=',orbplane2frame_(np.radians(10.0),np.radians(-31.124),np.radians(200.001)))

# print('drotz=',drotz(np.radians(10.0),np.radians(-31.124),np.radians(200.001)))

bbb= np.array( (np.radians(10.0),np.radians(-31.124),np.radians(200.001)) )
#bbb= np.array( (np.radians(0.0),np.radians(0.0),np.radians(0.0)) )
print('drotz2(bbb)=',drotz2( bbb ))

#print(dkep_r)
#print(kep_r_(1.0,0.1,np.radians(-29.99)))
#print(dkep_r(1.0,0.1,np.radians(-29.99)))

aaa = np.array((1.0,0.45,np.radians(1.0)))
# def printz(x):
#     return print(str(x)+' = ',x)
# printz(aaa)

print('aaa = ', aaa)
print('kep_r(aaa) = ',kep_r(aaa))

dkep_r = grad(kep_r)

print('dkep_r(aaa) = ',dkep_r(aaa))

print('xyz_orbplane(aaa) = ',xyz_orbplane(aaa))

dxyz_orbplane = jacobian(xyz_orbplane)

print('dxyz_orbplane(aaa) = ',dxyz_orbplane(aaa))

dxyz_orbplane_ = jacobian(xyz_orbplane_)

print('dxyz_orbplane_(aaa) = ',dxyz_orbplane_(aaa[0],aaa[1],aaa[2]))

ccc = np.array((aaa[0],aaa[1],aaa[2],bbb[0],bbb[1],bbb[2]))

print('xyz_frame(ccc) = ',xyz_frame(ccc))

print('xyz_frame_((aaa[0],aaa[1],aaa[2],bbb[0],bbb[1],bbb[2])) = ',xyz_frame((aaa[0],aaa[1],aaa[2],bbb[0],bbb[1],bbb[2])))

dxyz_frame = jacobian(xyz_frame)

print('dxyz_frame(ccc) = ',dxyz_frame(ccc))

print('meanmotion(1.0,1.1) = ', meanmotion(1.0,1.1))

mye = 0.3134123
myM = -21.98*np.pi
myEans = eccentricanomaly(mye,myM)
print('myEans-mye*np.sin(myEans)-np.mod(myM,2*np.pi) = ', myEans-mye*np.sin(myEans)-np.mod(myM,2*np.pi))

print('eccentricanomaly(mye,myM) = ', eccentricanomaly(mye,myM))

print('trueanomaly(0.1, 0.16*np.pi) = ', trueanomaly(0.1, 0.16*np.pi)/np.pi, '*pi')
print('trueanomaly(0.1, 0.56*np.pi) = ', trueanomaly(0.1, 0.56*np.pi)/np.pi, '*pi')
print('trueanomaly(0.1, 1.16*np.pi) = ', trueanomaly(0.1, 1.16*np.pi)/np.pi, '*pi')
print('trueanomaly(0.1, 1.56*np.pi) = ', trueanomaly(0.1, 1.56*np.pi)/np.pi, '*pi')
print('trueanomaly(0.1, 1.99*np.pi) = ', trueanomaly(0.1, 1.99*np.pi)/np.pi, '*pi')


