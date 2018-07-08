import numpy as np
from orbitdeterminator.util.new_tle_kep_state import kep_to_state

mu = 398600.4405
J2 = 1.08262668e-3
Re = 6378.137
we = 7.292115e-5
ee = 0.08181819

def drag(s):
    r = np.linalg.norm(s[0:3])
    v_atm = we*np.array([-s[1],s[0],0])
    v_rel = s[3:6] - v_atm

    rs = Re*(1-(ee*s[2]/r)**2)
    h = r-rs
    p = 0.6*np.exp(-(h-175)*(29.4-0.012*h)/915) # in kg/km^3
    coeff = 3.36131e-9     # in km^2/kg
    acc = -p*coeff*np.linalg.norm(v_rel)*v_rel

    return acc

def j2_pert(s):
    r = np.linalg.norm(s[0:3])
    K = -3*mu*J2*(Re**2)/2/r**5
    comp = np.array([1,1,3])
    comp = comp - 5*(s[2]/r)**2
    comp = np.multiply(comp,s[0:3])
    comp = np.multiply(K,comp)

    return comp

def sdot(s):
    mu = 398600.4405
    r = np.linalg.norm(s[0:3])
    a = -mu/(r**3)*s[0:3]

    p_j2 = j2_pert(s)
    p_drag = drag(s)

    a = a+p_j2+p_drag
    return np.array([*s[3:],*a])

def rkf45(s,t0,tf,h=10,tol=1e-6):
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

    return s

def propagate_state(s,t0,tf):
    return rk4(s,t0,tf)

def propagate_kep(kep,t0,tf):
    s = kep_to_state(kep)
    return rk4(s,t0,tf)

if __name__ == "__main__":
    #s = np.array([-1.25407719e+02,6.23615970e+03,2.67702463e+03,-5.21833653e+00,2.12226865e+00,-5.19435315e+00])
    s = np.array([2.87327861e+03,5.22872234e+03,3.23884457e+03,-3.49536799e+00,4.87267295e+00,-4.76846910e+00])
    t0, tf = 0, 88796.3088704
    final = propagate_state(s,t0,tf)
    print(final)
