"""Numerical orbit propagator based on RK4. Takes into account J2 and drag perturbations."""

import numpy as np

mu_Earth = 398600.4418  # gravitational parameter mu
J2 = 1.08262668e-3 # J2 coefficient
Re = 6378.137  # equatorial radius of the Earth
we = 7.292115e-5  # rotation rate of the Earth in rad/s
ee = 0.08181819  # eccentricity of the Earth's shape

def drag(s):
    """Returns the drag acceleration for a given state.

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]

       Returns:
           1x3 numpy array: the drag acceleration [ax,ay,az]
    """

    r = np.linalg.norm(s[0:3])
    v_atm = we*np.array([-s[1],s[0],0])   # calculate velocity of atmosphere
    v_rel = s[3:6] - v_atm

    rs = Re*(1-(ee*s[2]/r)**2)   # calculate radius of surface
    h = r-rs
    p = 0.6*np.exp(-(h-175)*(29.4-0.012*h)/915) # in kg/km^3
    coeff = 3.36131e-9     # in km^2/kg
    acc = -p*coeff*np.linalg.norm(v_rel)*v_rel

    return acc

def j2_pert(s):
    """Returns the J2 acceleration for a given state.

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]

       Returns:
           1x3 numpy array: the J2 acceleration [ax,ay,az]
    """

    r = np.linalg.norm(s[0:3])
    K = -3*mu_Earth*J2*(Re**2)/2/r**5
    comp = np.array([1,1,3])
    comp = comp - 5*(s[2]/r)**2
    comp = np.multiply(comp,s[0:3])
    comp = np.multiply(K,comp)

    return comp

def sdot(s):
    """Returns the time derivative of a given state.

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]

       Returns:
           1x6 numpy array: the time derivative of s [vx,vy,vz,ax,ay,az]
    """

    mu_Earth = 398600.4405
    r = np.linalg.norm(s[0:3])
    a = -mu_Earth/(r**3)*s[0:3]

    p_j2 = j2_pert(s)
    p_drag = drag(s)

    a = a+p_j2+p_drag
    return np.array([*s[3:6],*a])

def rkf45(s,t0,tf,h=10,tol=1e-6):
    """Runge-Kutta Fehlberg 4(5) Numerical Integrator

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]
           t0(float)  : initial time
           tf(float)  : final time
           h(float)   : step-size
           tol(float) : tolerance of error

      Returns:
           1x6 numpy array: the state at time tf
    """

    t = t0
    while(tf-t > 0.00001):
        if (tf-t < h):
            h = tf-t

        k1 = h*sdot(s)
        k2 = h*sdot(s+k1/4)
        k3 = h*sdot(s+3/32*k1+9/32*k2)
        k4 = h*sdot(s+1932/2197*k1-7200/2197*k2+7296/2197*k3)
        k5 = h*sdot(s+439/216*k1-8*k2+3680/513*k3-845/4104*k4)
        k6 = h*sdot(s-8/27*k1+2*k2-3544/2565*k3+1859/4104*k4-11/40*k5)

        y = s+25/216*k1+1408/2565*k3+2197/4104*k4-k5/5
        z = s+16/135*k1+6656/12825*k3+28561/56430*k4-9/50*k5+2/55*k6

        s = y
        t = t+h

        err = np.linalg.norm(y-z)
        h = h*0.84*(tol/err)**0.25

    return s

def rk4(s,t0,tf,h=30):
    """Runge-Kutta 4th Order Numerical Integrator

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]
           t0(float)  : initial time
           tf(float)  : final time
           h(float)   : step-size

      Returns:
           1x6 numpy array: the state at time tf
    """

    t = t0

    if tf < t0:
        h = -h

    while(abs(tf-t) > 0.00001):
        if (abs(tf-t) < abs(h)):
            h = tf-t

        k1 = h*sdot(s)
        k2 = h*sdot(s+k1/2)
        k3 = h*sdot(s+k2/2)
        k4 = h*sdot(s+k3)

        s = s+(k1+2*k2+2*k3+k4)/6
        t = t+h

 #       if (s[2]<0 and s[2]>-200 and s[5]>0):
 #           dt = -s[2]/s[5]
 #           print(t+dt)

    return s

def time_period(s,h=30):
    """Returns the nodal time period of an orbit.

       Args:
           s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]
           h(float): step-size

       Returns:
           float: the nodal time period of the orbit
    """

    t = 0

    old_z, pass_1 = 0, None

    while(True):
        k1 = h*sdot(s)
        k2 = h*sdot(s+k1/2)
        k3 = h*sdot(s+k2/2)
        k4 = h*sdot(s+k3)

        s = s+(k1+2*k2+2*k3+k4)/6
        t = t+h

        if (s[2]>=0 and old_z<0):
            dt = -s[2]/s[5]
            t2 = t+dt

            if pass_1 is None:
                pass_1 = t2
            else:
                return t2-pass_1

        old_z = s[2]

def propagate_state(s,t0,tf):
    """Equivalent to the rk4 function."""

    return rk4(s,t0,tf)

if __name__ == "__main__":
    s = np.array([2.87393871e+03,5.22992358e+03,3.23958865e+03,-3.49496655e+00,4.87211332e+00,-4.76792145e+00])
    t0, tf = 0, 88796.3088704
    final = propagate_state(s,t0,tf)
    print(final)
