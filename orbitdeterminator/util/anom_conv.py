"""Vectorized anomaly conversion scripts"""

import numpy as np

def true_to_ecc(theta,e):
    """Converts true anomaly to eccentric anomaly.

       Args:
           theta(numpy array): array of true anomalies (in radians)
           e(float): eccentricity

       Returns:
           numpy array: array of eccentric anomalies (in radians)
    """

    pi2=2*np.pi
    raw = np.arctan2((1-e**2)*np.sin(theta), e+np.cos(theta))
    raw = raw%pi2

    # revolution correction code
    correct = (theta//pi2)*pi2
    raw = raw+correct
    correct = np.clip((raw-theta)//pi2,0,None)*pi2

    return raw-correct

def ecc_to_mean(E,e):
    """Converts eccentric anomaly to mean anomaly.

       Args:
           E(numpy array): array of eccentric anomalies (in radians)
           e(float): eccentricity

       Returns:
           numpy array: array of mean anomalies (in radians)
    """

    return E - e*np.sin(E)

def mean_to_t(M,a):
    """Converts mean anomaly to time elapsed.

       Args:
           M(numpy array): array of mean anomalies (in radians)
           a(float): semi-major axis

       Returns:
           numpy array: numpy array of time elapsed
    """

    n = (a**3/398600.4405)**0.5
    return n*(M-M[0])

if __name__ == "__main__":
    thetas = np.linspace(0,4*np.pi,21)
    e = 0.7151443
    print(thetas)
    ecc = true_to_ecc(thetas,e)
    print(ecc)
    mean = ecc_to_mean(ecc,e)
    print(mean)
    t = mean_to_t(mean,26505.1836)
    print(t)
