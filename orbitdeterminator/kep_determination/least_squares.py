"""Computes the least-squares optimal Keplerian elements for a sequence of
   cartesian position observations.
"""

import math
import numpy as np
#Thinly-wrapped numpy
#import autograd.numpy as np
import matplotlib.pyplot as plt
# from autograd import grad
# from autograd import jacobian
# from autograd import elementwise_grad as egrad

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

# compute true anomaly (f) from eccentricity (e) and eccentric anomaly (E)
def trueanomaly(e,E):
    Enew = np.mod(E,2.0*np.pi)
    return 2.0*np.arctan(  np.sqrt((1.0+e)/(1.0-e))*np.tan(Enew/2)  )

# compute eccentric anomaly (E) from eccentricity (e) and true anomaly (f)
def truean2eccan(e, f):
    fnew = np.mod(f,2.0*np.pi)
    return 2.0*np.arctan(  np.sqrt((1.0-e)/(1.0+e))*np.tan(fnew/2)  )

# compute true anomaly from eccentricity and mean anomaly
def meanan2truean(e,M):
    return trueanomaly(e, eccentricanomaly(e, M))

# compute true anomaly from time, a, e, mu and taup
def time2truean(a, e, mu, t, taup):
    return meanan2truean(e, meananomaly(meanmotion(mu, a), t, taup))

# compute cartesian positions (x,y,z) at time t
# for mass parameter mu from orbital elements a, e, taup, I, omega, Omega
def orbel2xyz(t, mu, a, e, taup, omega, I, Omega):

    # compute true anomaly at time t
    f = time2truean(a, e, mu, t, taup)
    # get cartesian positions wrt inertial frame from orbital elements
    return xyz_frame_(a, e, f, omega, I, Omega)

# # TODO:
# # write function to compute range as a function of orbital elements: DONE
# # write function to compute true anomaly as a function of time-of-fly: DONE
# # the following transformation is needed: from time t, to mean anomaly M,
# # to eccentric anomaly E, to true anomaly f, i.e.:
# # t -> M=n*(t-taup) -> M=E-e*sin(E) (invert) ->
# # -> f = 2*atan(  sqrt((1+e)/(1-e))*tan(E/2)  ): DONE
# # write function which takes observed values and computes the difference wrt expected-to-be-observed values as a function of unknown orbital elements (to be fitted): DONE
# # compute Q as a function of unknown orbital elements (to be fitted): DONE
# # optimize Q -> return fitted orbital elements (requires an ansatz: take input from minimalistic Gibb's?)

# NOTES:
# matrix multiplication of numpy's 2-D arrays is done through `np.matmul`

data = np.loadtxt('../orbit.csv',skiprows=1,usecols=(0,1,2,3))

print('data[0,:] = ', data[0,:])

print('data.shape = ', data.shape)
print('data.shape[0] = ', data.shape[0])

#Earth's mass parameter
mu_Earth = 398600.435436E9 # m^3/seg^2 # 398600.435436 # km^3/seg^2

a_ = 6801088.421358589 # m
e_ = 0.000994284676986928
I_ = np.deg2rad(51.64073790913945) #deg
omega_ = np.deg2rad(111.46902673189568) #deg
Omega_ = np.deg2rad(112.51570524695879) #deg
f_ = np.deg2rad(248.67209974376843) #deg

E_ = truean2eccan(e_, f_)
M_ = E_-e_*np.sin(E_)
n_ = meanmotion(mu_Earth,a_)
taup_ = data[0,0]-M_/n_

print('taup_ = ', taup_)
print('n_ = ', n_)
print('T_ = 2pi/n_ = ', 2.0*np.pi/n_)
print('n_*(t0-taup_) = ', n_*(data[0,0]-taup_)/np.pi, '*pi')
print('M_ = ', M_)

my_xyz_ = orbel2xyz(data[0,0], mu_Earth, a_, e_, taup_, omega_, I_, Omega_)
print('orbel2xyz(t, mu, a, e, taup, omega, I, Omega) = ', my_xyz_ )
print('r(x,y,z) = ',  np.linalg.norm(my_xyz_, ord=2))

# x0 = np.array((a_, e_, data[1719,0], omega_, I_, Omega_))
x0 = np.array((a_, e_, taup_, omega_, I_, Omega_))

print('x0 = ', x0)

#the arithmetic mean will be used as the reference epoch for the elements
# t_mean = np.mean(data[:,0])

# print('t_mean = ', t_mean)

# y = data[0,1:4] - orbel2xyz(data[0,0], mu_Earth, x0[0], x0[1], x0[2], x0[3], x0[4], x0[5])

# print('y = ', y)
# print('np.linalg.norm(y,ord=2) = ', np.linalg.norm(y,ord=2))

ranges_ = np.sqrt(data[:,1]**2+data[:,2]**2+data[:,3]**2)

print('ranges_[0:10] = ', ranges_[0:10])

def Q(x,my_data,my_mu_Earth):
    Q0 = 0.0
    for i in range(0,my_data.shape[0]-1):
        #initializing residuals vector
        #print('all_residuals = ', all_residuals)
        #all_residuals = np.zeros(data.shape[0])
        Q0 = Q0 + np.linalg.norm(my_data[i,1:4] - orbel2xyz(my_data[i,0], my_mu_Earth, x[0], x[1], x[2], x[3], x[4], x[5]), ord=2)/my_data.shape[0]
    return Q0

#print('Q(x0, data, mu_Earth) = ', Q(x0, data[0:50,:], mu_Earth))

def QQ(x):
    return Q(x, data[0:200,:], mu_Earth)

print('Total residual evaluated at initial guess: ', QQ(x0))

from scipy.optimize import minimize

Q_mini = minimize(QQ,x0,method='nelder-mead',options={'maxiter':50})

# print('Q_mini.x = ', Q_mini.x)
# print('Q_mini.x-x0 = ', Q_mini.x-x0)

print('Total residual evaluated at least-squares solution: ', QQ(Q_mini.x))

import matplotlib.pyplot as plt

plt.scatter( data[:,0], ranges_ ,s=0.1)
plt.plot( data[:,0], kep_r_(x0[0], x0[1], time2truean(x0[0], x0[1], mu_Earth, data[:,0], x0[2])), color="green")
plt.plot( data[:,0], kep_r_(Q_mini.x[0], Q_mini.x[1], time2truean(Q_mini.x[0], Q_mini.x[1], mu_Earth, data[:,0], Q_mini.x[2])), color="orange")
plt.xlabel('time')
plt.ylabel('range')
plt.title('Fit vs observations: range')
plt.show()

