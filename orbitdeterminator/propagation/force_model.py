import time

import numpy as np

from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv

avg_bstar = 0.21109E-4

def propagate_kep(kep,init_time,final_time,bstar=avg_bstar):
    t0 = time.gmtime(init_time)

    t0 = ((t0.tm_year%100)*1000 +
           t0.tm_yday+
           t0.tm_hour/24 + t0.tm_min/1440 + t0.tm_sec/86400)

    t0 = "{:14.8f}".format(t0)
    tf = time.gmtime(final_time)

    mu = 398600.4405
    n = 86400/2/np.pi * (mu/kep[0]**3)**0.5

    tanom = np.radians(kep[5])
    e = kep[1]
    ecc = np.arctan2((1-e**2)**0.5*np.sin(tanom),e+np.cos(tanom))
    ecc = ecc%(2*np.pi)
    mean = ecc - e*np.sin(ecc)
    mean = np.degrees(mean)

    inc  = "{:8.4f}".format(kep[2])
    raan = "{:8.4f}".format(kep[4])
    e = "{:.7f}".format(e)[2:]
    argp = "{:8.4f}".format(kep[3])        
    mean = "{:8.4f}".format(mean)
    n = "{:11.8f}".format(n)

    bexp = np.floor(np.log10(abs(bstar)))+1
    bstar = "{:+5}{:+.0f}".format(int(bstar*10**(-bexp+5)),bexp)

    line1 = ('1 00000U 000000   '+t0+'  '
             '.00000000  00000-0 '+bstar+' 0  0000')
    line2 = ('2 00000 '+inc+' '+raan+' '+e+' '+argp+
             ' '+mean+' '+n+'000000')
    
    satellite = twoline2rv(line1, line2, wgs72)
    position, velocity = satellite.propagate(
        tf.tm_year, tf.tm_mon, tf.tm_mday, tf.tm_hour, tf.tm_min, tf.tm_sec)
    
    return position,velocity

if __name__ == "__main__":
    t0 = 1526927274
    tf = 1526932833
    
    kep = np.array([6782.96, 0.0004084, 51.6402, 108.2140, 150.4026, 238.0528])
    
    pos, vel = propagate_kep(kep,t0,tf)
    print(pos)
    print(vel)
