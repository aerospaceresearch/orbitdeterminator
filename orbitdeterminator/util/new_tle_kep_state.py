import math
import numpy as np

def MtoE(M,e):
    E = M
    dy = 1
    while(abs(dy) > 0.0001):
        M2 = E - e*math.sin(E)
        dy = M - M2
        dx = dy/(1-e*math.cos(E))
        E = E+dx

    return E

def kep_to_state(kep):
    ''' # this function uses the keplerian elements to compute the position and velocity vector

	# input

	# kep is a 1x6 matrix which contains the following variables
	# kep(0)=inclination (degrees)
	# kep(1)=right ascension of the ascending node (degrees)
	# kep(2)=eccentricity (number)
	# kep(3)=argument of perigee (degrees)
	# kep(4)=mean anomaly (degrees)
	# kep(5)=mean motion (revs per day)

	# output

	# r = 1x6 matrix which contains the position and velocity vector
	# r(0),r(1),r(2) = position vector (rx,ry,rz) m
	# r(3),r(4),r(5) = velocity vector (vx,vy,vz) m/s
        '''

    r = np.zeros((6,1))
    mu = 398600.4405

    # unload orbital elements array

    sma_pre = (398600.4405 * (86400 ** 2)) / ((kep[5] ** 2) * 4 * (math.pi ** 2));
    sma = sma_pre ** (1.0 / 3.0)  # sma is semi major axis, we use mean motion (kep(6)) to compute this
    ecc = kep[2]  # eccentricity
    inc = math.radians(kep[0])  # inclination
    argp = math.radians(kep[3])  # argument of perigee
    raan = math.radians(kep[1])  # right ascension of the ascending node
    eanom = MtoE(math.radians(kep[4]), ecc)  # we use mean anomaly(kep(5)) and the function MtoE to compute eccentric anomaly (eanom)

    smb = sma * math.sqrt(1-ecc**2)

    x = sma * (math.cos(eanom) - ecc)
    y = smb * math.sin(eanom)

    # calculate position and velocity in orbital frame
    m_dot = (mu/sma**3)**0.5
    e_dot = m_dot/(1 - ecc*math.cos(eanom))
    x_dot = -sma * math.sin(eanom) * e_dot
    y_dot =  smb * math.cos(eanom) * e_dot

    # rotate them by argp degrees
    x_rot = x * math.cos(argp) - y * math.sin(argp)
    y_rot = x * math.sin(argp) + y * math.cos(argp)
    x_dot_rot = x_dot * math.cos(argp) - y_dot * math.sin(argp)
    y_dot_rot = x_dot * math.sin(argp) + y_dot * math.cos(argp)

    # convert them into 3D coordinates
    r[0] = x_rot * math.cos(raan) - y_rot * math.sin(raan) * math.cos(inc)
    r[1] = x_rot * math.sin(raan) + y_rot * math.cos(raan) * math.cos(inc)
    r[2] = y_rot * math.sin(inc)

    r[3] = x_dot_rot * math.cos(raan) - y_dot_rot * math.sin(raan) * math.cos(inc)
    r[4] = x_dot_rot * math.sin(raan) + y_dot_rot * math.cos(raan) * math.cos(inc)
    r[5] = y_dot_rot * math.sin(inc)

    return r

if __name__ == "__main__":

	kep = np.array([101.7540, 195.7370, 0.0031531, 352.8640, 117.2610, 12.53984625169364])

	r = kep_to_state(kep)
	print(r)
