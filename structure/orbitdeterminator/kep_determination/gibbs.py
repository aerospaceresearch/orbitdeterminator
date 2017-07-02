'''
Created by Alexandros Kazantzidis
Date : 2/07/17

Gibbs: Takes a positional data set in the format of (time, x, y, z) and produces one set of six keplerian
elements (a, e, i, ω, Ω, v) using Gibb's solution for preliminary orbit determination and Kalman filters
'''


import numpy as np
from math import *
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from util import read_data
import pandas as pd
pd.set_option('display.width', 1000)
from filters import sav_golay
from filters import triple_moving_average
from kep_determination import lamberts_kalman


def mtov(m, e):
    '''Computes true anomaly v from a given mean anomaly M and eccentricity e by using Newton-Raphson method

    Args:
        M(float) = mean anomaly 
        e(float) = eccentricity 

    Returns:
        v(float) = true anomaly 
    '''

    i = 1
    Eo = 100
    while True:
        E = Eo - ((Eo - e * sin(Eo) - m) / (1 - e * cos(Eo)))
        Eo = E
        i = i + 1
        if i == 100: break

    v_pre = (cos(E) - e) / (1 - e * cos(E))
    v_pre = radians(v_pre)
    v = acos(v_pre)
    v = degrees(v)
    return v


def kep_gibbs(x1, x2, x3):
    '''Takes three position points - numpy arrays with time,x,y,z as elements
       and produces the keplerian elements of the orbit these three points shape using Gibbs solution

    Args:
        x1(numpy array): time and position for point 1 [time1,x1,y1,z1]
        x2(numpy array): time and position for point 2 [time2,x2,y2,z2]
        x3(numpy array): time and position for point 2 [time2,x2,y2,z2]

    Returns:
        kep(numpy array) : a numpy array containing the keplerian elements computed for the three points given
                            [semi major axis (a), eccentricity (e), inclination (i), argument of perigee (ω),
                            right ascension of the ascending node (Ω), true anomaly (v)] format
    '''

    # Define constants
    mu = 398600.4418
    pi = 3.141592653


    # Calculate vector magnitudes and cross products

    r1 = np.linalg.norm(x1)
    r2 = np.linalg.norm(x2)
    r3 = np.linalg.norm(x3)

    c12 = np.cross(x1,x2)
    c23 = np.cross(x2,x3)
    c31 = np.cross(x3,x1)

    #Calculate D and N vectors

    N = r1*c23 + r2*c31 + r3*c12
    D = c12 + c23 + c31

    # Check for sanity


    # Calcualte P, from pD = N

    p = np.linalg.norm(N)/np.linalg.norm(D)

    # Calculate S, then find e using e = S/D

    S = x1*(r2 - r3) + x2*(r3 - r1) + x3*(r1 - r2)
    e = np.linalg.norm(S)/np.linalg.norm(D)

    # Q = w x r, calculate w then find Q

    W = N/np.linalg.norm(N)
    Q = S/np.linalg.norm(S)

    # Calculate B and L

    B = np.cross(D,x2);
    L = sqrt(mu/(np.linalg.norm(D) * np.linalg.norm(N)))

    # Calculate V2

    v2 = (np.linalg.norm(L)/r2)*B + np.linalg.norm(L) * S

    # Using R2 and v2, calculate orbital elements

    H = np.cross(x2,v2)

    #Calculate i (inclination)

    inclination = acos(H[2]/np.linalg.norm(H))
    inclination = inclination * 180/pi

    n = [-H[1], H[0], 0]

    v=np.linalg.norm(v2)
    r=np.linalg.norm(x2)

    vr = np.dot(x2,v2)/r

    E = 1/mu*((v*v - mu/r)*x2 - np.dot(x2,v2)*v2)

    #Calculate Omega (longitude of the ascending node)

    Omega = acos(n[0]/np.linalg.norm(n)) * 180/pi

    #Calculate omega (argument of periapsis)

    omega = acos(np.dot(n,E)/(np.linalg.norm(n) * np.linalg.norm(E))) * 180/pi

    #Calculate nu (mean anomaly)

    nu = acos(np.dot(E,x2)/(np.linalg.norm(E) * np.linalg.norm(x2))) * 180/pi

    v = mtov(nu, e)

    #Calculate eccentricity

    e = np.linalg.norm(E)

    # Calculate orbital major axis

    a = p / (1-(e*e))

    result = [a, e, inclination, omega, Omega, v]
    return result


def create_kep(my_data):
    '''Computes all the keplerian elements for every point of the orbit you provide using Gibbs method


        Args:
            my_data(csv file) : read file csv that contains the positional data set in (Time, x, y, z) Format


        Returns:
            kep(numpy array) : a numpy array containing all the keplerian elements computed for the orbit given in
                               [semi major axis (a), eccentricity (e), inclination (i), argument of perigee (ω),
                               right ascension of the ascending node (Ω), true anomaly (v)] format
    '''

    kep_final = np.zeros(((len(my_data)-3), 6))
    for i in range(0, (len(my_data) - 3)):

        kep_final[i, :] = kep_gibbs(my_data[i, 1:4], my_data[i+1, 1:4], my_data[i+2, 1:4])

    return kep_final


if __name__ == "__main__":

    my_data = read_data.load_data("orbit.csv")
    my_data = triple_moving_average.generate_filtered_data(my_data, 3)
    my_data = sav_golay.golay(my_data, 101, 6)
    kep_final = create_kep(my_data)
    kep_final = lamberts_kalman.kalman(kep_final, 0.01 ** 2)
    print(pd.DataFrame(kep_final))
