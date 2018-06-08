"""Computes the least-squares optimal Keplerian elements for a sequence of
   cartesian position observations.
"""

# # DEVELOPMENT ROADMAP:
# # write function to compute range as a function of orbital elements: DONE
# # write function to compute true anomaly as a function of time-of-fly: DONE
# # the following transformation is needed: from time t, to mean anomaly M,
# # to eccentric anomaly E, to true anomaly f, i.e.:
# # t -> M=n*(t-taup) -> M=E-e*sin(E) (invert) ->
# # -> f = 2*atan(  sqrt((1+e)/(1-e))*tan(E/2)  ): DONE
# # write function which takes observed values and computes the difference wrt expected-to-be-observed values as a function of unknown orbital elements (to be fitted): DONE
# # compute Q as a function of unknown orbital elements (to be fitted): DONE
# # optimize Q -> return fitted orbital elements (requires an ansatz: take input from minimalistic Gibb's?)

# NOTES to self:
# matrix multiplication of numpy's 2-D arrays is done through `np.matmul`

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# evaluate cost function given a set of observations
def Q(x,my_data,my_mu_Earth):
    Q0 = 0.0
    for i in range(0,my_data.shape[0]-1):
        # observed xyz values
        xyz_obs = my_data[i,1:4]
        # predicted )computed xyz values
        xyz_com = orbel2xyz(my_data[i,0], my_mu_Earth, x[0], x[1], x[2], x[3], x[4], x[5])
        # observed minus computed residual:
        xyz_res = xyz_obs-xyz_com
        #square residual, add to total cost function, divide by number of observations
        Q0 = Q0 + np.linalg.norm(xyz_res, ord=2)/my_data.shape[0]
    return Q0

#########################

# Earth's mass parameter in appropriate units:
mu_Earth = 398600.435436E9 # m^3/seg^2

# load observational data:
data = np.loadtxt('../orbit.csv',skiprows=1,usecols=(0,1,2,3))

# construct cost function of only one argument, x
# due to optimization of processing time, only the first 800 data points are used
# nevertheless, this is enough to improve the solution
def QQ(x):
    return Q(x, data[0:800,:], mu_Earth)


# generate vector of initial guess of orbital elements:
# values written below correspond to solution of ellipse_fit.py for the same file
a_ = 6801088.421358589 # m
e_ = 0.000994284676986928
I_ = np.deg2rad(51.64073790913945) #deg
omega_ = np.deg2rad(111.46902673189568) #deg
Omega_ = np.deg2rad(112.51570524695879) #deg
f_ = np.deg2rad(248.67209974376843) #deg

E_ = truean2eccan(e_, f_) #ecc. anomaly
M_ = E_-e_*np.sin(E_) #mean anomaly
n_ = meanmotion(mu_Earth,a_) #mean motion
taup_ = data[0,0]-M_/n_ #time of pericenter passage

# this is the vector of initial guess of orbital elements:
x0 = np.array((a_, e_, taup_, omega_, I_, Omega_))

print('Orbital elements, initial guess:')
print('Semi-major axis (a):                 ',a_,'m')
print('Eccentricity (e):                    ',e_)
print('Time of pericenter passage (tau):    ',taup_,'sec')
print('Argument of pericenter (omega):      ',np.rad2deg(omega_),'deg')
print('Inclination (I):                     ',np.rad2deg(I_),'deg')
print('Longitude of Ascending Node (Omega): ',np.rad2deg(Omega_),'deg')

#the arithmetic mean will be used as the reference epoch for the elements
t_mean = np.mean(data[:,0])

# the observed range as a function of time will be used for plotting
ranges_ = np.sqrt(data[:,1]**2+data[:,2]**2+data[:,3]**2)

# minimize cost function QQ, using initial guess x0
Q_mini = minimize(QQ,x0,method='nelder-mead',options={'maxiter':100, 'disp': True})

#display least-squares solution
print('Orbital elements, least-squares solution:')
print('Reference epoch (t0):                ', t_mean)
print('Semi-major axis (a):                 ', Q_mini.x[0], 'm')
print('Eccentricity (e):                    ', Q_mini.x[1])
print('Time of pericenter passage (tau):    ', Q_mini.x[2], 'sec')
print('Argument of pericenter (omega):      ', np.rad2deg(Q_mini.x[3]), 'deg')
print('Inclination (I):                     ', np.rad2deg(Q_mini.x[4]), 'deg')
print('Longitude of Ascending Node (Omega): ', np.rad2deg(Q_mini.x[5]), 'deg')

print('Total residual evaluated at initial guess: ', QQ(x0))
print('Total residual evaluated at least-squares solution: ', QQ(Q_mini.x))
print('Percentage improvement: ', (QQ(x0)-QQ(Q_mini.x))/QQ(x0)*100, ' %')

#generate plots:
plt.scatter( data[:,0], ranges_ ,s=0.1)
plt.plot( data[:,0], kep_r_(x0[0], x0[1], time2truean(x0[0], x0[1], mu_Earth, data[:,0], x0[2])), color="green")
plt.plot( data[:,0], kep_r_(Q_mini.x[0], Q_mini.x[1], time2truean(Q_mini.x[0], Q_mini.x[1], mu_Earth, data[:,0], Q_mini.x[2])), color="orange")
plt.xlabel('time')
plt.ylabel('range')
plt.title('Fit vs observations: range')
plt.show()

