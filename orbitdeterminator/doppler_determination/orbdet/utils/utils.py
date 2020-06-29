import numpy as np

from scipy.integrate import odeint
from sgp4.api import Satrec
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, ITRS, ICRS, TEME, CartesianDifferential, CartesianRepresentation

from orbdet.utils.constants import *

def range_range_rate(x_sat:np.ndarray, x_obs:np.ndarray):
    """ Get range and slant range rate (radial relative velocity component). 
        Vectorized.

    Args:
        x_sat (np.ndarray): satellite location (pos, vel).
        x_obs (np.ndarray): observer location (pos, vel).
    Returns:
        r (np.ndarray): range.
        rr (np.ndarray): range rate (slant range rate).
    """

    if len(x_obs.shape) == 2:   # Single observer   (6, n)
        einsum_format = 'ij,ij->j'
        d = x_sat - x_obs   # Difference
    elif len(x_obs.shape) == 3: # Multiple observers    (6,n,n_obs)
        einsum_format = 'ijk,ijk->jk'
        d = np.repeat(np.expand_dims(x_sat, axis = 2), x_obs.shape[2], axis = 2) - x_obs    # Difference

    #d = x_sat - x_obs                       # Difference
    r = np.linalg.norm(d[0:3,], axis=0)               # Range
    l = d[0:3,] / np.linalg.norm(d[0:3,], axis=0)     # Range unit vectors
    rr = np.einsum(einsum_format, d[3:6,], l)         # Radial range rate  

    return r.T, rr.T

def doppler_shift(x_sat:np.ndarray, x_obs:np.ndarray, f_ref:float, c:float):
    """ Get Doppler shift value for the give satellite and observer vectors.
        Vectorized.
    
    Args:
        x_sat (np.ndarray): satellite location (pos, vel).
        x_obs (np.ndarray): observer location (pos, vel).
        f_ref (float): reference frequency.
        c (float): propagation speed.
    
    Returns:
        df (np.ndarray): frequency shift relative to reference frequenct df
    """

    _, rv = range_range_rate(x_sat, x_obs)
    df = rv / c * f_ref

    return df

# Orbit derivative
def orbdyn_2body(x:np.ndarray, t:float, mu:float=3.986004418e14):
    """ Orbital (x,y,z,x_dot,y_dot,z_dot) vector derivative

    Args:
        x (np.ndarray): state vector.
        t (float): time.
    Returns:
        dxdt (np.ndarray): state vector time derivative.
    """

    r = np.linalg.norm(x[0:3,], axis=0) 

    dxdt = np.zeros(x.shape) 
    dxdt[0:3,] = x[3:6,]
    dxdt[3:6,] = -(mu/r**3) * x[0:3,]

    return dxdt

def orbdyn_2body_stm(x:np.ndarray, t:float, mu:float=3.986004418e14):
    """ Orbital (x,y,z,x_dot,y_dot,z_dot) vector and matrix derivative.
        Phi_dot = A * Phi. 

    Args:
        x (np.ndarray): state vector and flattened state transition matrix [x, Phi(:)]
                        Size: (6+6*6,): (42,).
                        
        t (float): time.
    Returns:
        dxdt (np.ndarray): state vector and state transition matrix time derivative.
    """

    dxdt = np.zeros(x.shape) 

    r = np.linalg.norm(x[0:3,], axis=0)
    dxdt[0:3,] = x[3:6,]
    dxdt[3:6,] = (-mu / r**3) * x[0:3,]
    A = get_matrix_A(x[0:3,], mu=mu)    # (6,6,n)
    
    if len(x.shape) == 1:
        Phi = x[6:,].reshape((6, 6))    # (6,6)
        Phi_dot = np.matmul(A, Phi)
        dxdt[6:,] = Phi_dot.reshape((36))
    else:
        Phi = x[6:,].reshape((6, 6, -1))    # (6,6,n)
        Phi_dot = np.einsum('ijl,jkl->ikl', A, Phi)
        dxdt[6:,] = Phi_dot.reshape((36, -1))

    return dxdt

def get_matrix_A(x:np.ndarray, mu:float=3.986004418e14):
    """ Get A matrix (orbital x_dot = A*x). Vectorized.

    Args:
        x (np.ndarray):  orbital state vector (Cartesian).
        mu (np.ndarray): standard gravitational parameter. Defaults to 3.98e14 m^3/s^2.
    Returns:
        A (np.ndarray): A matrix. Size (x_dim, x_dim): (6,6).
    """

    r = np.linalg.norm(x[0:3,], axis=0)
    aa = -mu / r**3
    b = 3 * mu / r**5

    AA = np.array([
        [aa + b*x[0,]**2, b*x[0,]*x[1,], b*x[0,]*x[2,]],
        [b*x[0,]*x[1,], aa + b*x[1,]**2, b*x[1,]*x[2,]],
        [b*x[0,]*x[2,], b*x[1,]*x[2,], aa + b*x[2,]**2,]
    ])   
    
    A_z = np.zeros(AA.shape)    # Zero parts for A matrix
    A_e = np.zeros(AA.shape)    # Eye (upper right)

    i = np.arange(AA.shape[0])
    A_e[i, i, ] = 1

    A = np.concatenate([
        np.concatenate([A_z, A_e], axis=1), 
        np.concatenate([AA,  A_z], axis=1)
    ], axis=0)

    return A

def f_obs_range_rate(x_sat:np.ndarray, x_obs:np.ndarray):
    """ Observation function for range rate.

    Args:
        x_sat (np.ndarray): set of satellite positions.
        x_obs (np.ndarray): set of observer positions.
    Returns:
        rr (np.ndarray): range rate. Size (z_dim, n): (1, n)
        H (np.ndarray): Partial of radial range rate w.r.t state vector.
                        Size (z_dim, x_dim, n): (1, 6, n).
    """

    _, rr = range_range_rate(x_sat, x_obs)
    H = get_matrix_range_rate_H(x_sat, x_obs)

    if len(x_obs.shape) == 2:
        rr = np.expand_dims(rr, axis=0)

    return rr, H

def f_obs_x_sat(x_sat:np.ndarray, x_obs:np.ndarray=None):
    """ Observation function for full state vector.
        E.g. GPS measurement

        Used for debugging.
    
    Args:
        x_sat (np.ndarray): set of satellite positions.
    Returns:
        x_sat (np.ndarray): satellite state vector.
        H   (np.ndarray): observation matrix (identity).
    """

    H = np.expand_dims(np.eye(x_sat.shape[0]), axis=2)
    H = np.repeat(H, x_sat.shape[1], axis=2)

    return x_sat, H

def get_matrix_range_rate_H(x_sat:np.ndarray, x_obs:np.ndarray):
    """ Obtain measurement Jacobian for range rate measurements. Vectorized.

    Args:
        x_sat (np.ndarray): set of satellite positions.
        x_obs (np.ndarray): set of observer positions.
    Returns:
        H (np.ndarray): Partial of radial range rate w.r.t state vector.
                        Size (z_dim, x_dim, n): (1, 6, n).
    """

    if len(x_obs.shape) == 2:   # Single observer   (6, n)
        einsum_format = 'ij,ij->j'
        d = x_sat - x_obs   # Difference
    elif len(x_obs.shape) == 3: # Multiple observers    (6,n,n_obs)
        einsum_format = 'ijk,ijk->jk'
        d = np.repeat(np.expand_dims(x_sat, axis = 2), x_obs.shape[2], axis = 2) - x_obs    # Difference

    #d = x_sat - x_obs                       # Difference
    r = np.linalg.norm(d[0:3,], axis=0)     # Range
    d_r = d / r                             # Temporary variable

    H = d_r[[3,4,5,0,1,2],]
    r_dot_v = np.einsum(einsum_format, d[0:3,], d[3:6])      # Dot product position, velocity
    H[0:3,:] -= (d[0:3,] * r_dot_v) / r**3

    if len(x_obs.shape) == 2:   # Single observer   (6, n)
        H = np.expand_dims(H, axis=0)
    elif len(x_obs.shape) == 3: # Multiple observers    (6,n,n_obs)
        H = np.transpose(H, (2, 0, 1))
    return H      # Transpose before return (H is a single row matrix)

def batch(
    x_0: np.ndarray, 
    P_bar_0: np.ndarray, 
    R: np.ndarray, 
    z: np.ndarray, 
    t: np.ndarray, 
    x_obs: np.ndarray, 
    f_obs, 
    tolerance: float = 1e-8,
    max_iterations: int = 1000
):
    """ Batch estimation algorithm. 

    Reference:  B. D. Tapley, B. E. Schultz, G. H. Born - Statistical Orbit Determination, 
                Chapter 4.6, p. 196-197 - Computational Algorithm for the Batch Processor.

    Args:
        x_0 (np.ndarray): Initial state vector, shape (x_dim, 1).
        P_bar_0 (np.ndarray): Initial uncertainty, shape (x_dim, x_dim).
        R (np.ndarray): Measurement uncertainty, shape (z_dim, z_dim).
        z (np.ndarray): Array of measurements, shape (z_dim, n).
        t (np.ndarray): Array of time deltas, shape (n,).
        x_obs (np.ndarray): Array of observer positions (x_dim, n).
        f_obs (): observation function.
        tolerance (float): convergence tolerance.

    Return:
        x_0 (np.ndarray): new estimate for the initial state vector.
    """
    n = z.shape[1]

    Phi_0 = np.eye(x_0.shape[0])    # Initial State Transition Matrix
    x_hat_0 = np.zeros(x_0.shape)   # Nominal trajectory update
    x_bar_0 = np.zeros(x_0.shape)   # Apriori estimate
    W = np.linalg.inv(R)

    W_vec = np.repeat(np.expand_dims(W, axis=2), n, axis=2)      

    error = 1
    i = 0

    while(np.abs(error) > tolerance and i < max_iterations):
        i += 1

        # Check if initial uncertainty has been set up
        if np.count_nonzero(P_bar_0) == 0:
            L = np.zeros(x_0.shape[0], x_0.shape[0])  
        else:
            L = np.linalg.inv(P_bar_0)

        N = L.dot(x_bar_0)

        # Propagate, flatten the stm and append to the state vector
        x_Phi = np.transpose(odeint(orbdyn_2body_stm, 
            np.concatenate([x_0.squeeze(), Phi_0.flatten()]), t, args=(MU,)))
        X = x_Phi[0:6,]
        Phi = x_Phi[6:,].reshape((x_0.shape[0], x_0.shape[0],  t.shape[0]))
        
        # Calculate projected observations (projected measurements and H_tilde)
        y, H_t = f_obs(X, x_obs)
        dy = np.expand_dims(z - y, axis=1)

        # Calculate H
        H_k = np.einsum('ijl,jkl->ikl', H_t, Phi)
        H_kt = np.transpose(H_k, axes=(1,0,2))

        # Batch update
        L += np.einsum('ijl,jkl,kml->im', H_kt, W_vec, H_k)
        N += np.einsum('ijl,jkl,kml->im', H_kt, W_vec, dy)

        temp = np.copy(x_hat_0)

        x_hat_0 = np.linalg.inv(L).dot(N)

        x_0 += + x_hat_0
        x_bar_0 -= x_hat_0

        error = np.abs(np.linalg.norm(temp - x_hat_0))

        np.set_printoptions(precision=2)

    output = {'num_it': i}

    return x_0, output