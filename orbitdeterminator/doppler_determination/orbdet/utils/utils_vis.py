import numpy as np
import matplotlib.pyplot as plt

from orbdet.utils.constants import *
from orbdet.utils.utils import *

from mpl_toolkits.mplot3d import Axes3D

def plot_sphere(ax, d:np.ndarray, n:np.ndarray) -> None:
    """ Plots a sphere on a given axes object.

    Args:
        ax (matplotlib.axes): axes to plot on
        d (float): sphere diameter
        n (float): grid resolution
    """
    
    u = np.linspace(0, np.pi, n)
    v = np.linspace(0, 2 * np.pi, n)

    x = d * np.outer(np.sin(u), np.sin(v))
    y = d * np.outer(np.sin(u), np.cos(v))
    z = d * np.outer(np.cos(u), np.ones_like(v))

    ax.plot_wireframe(x, y, z, alpha=0.25, linewidth=1, color='gray')

def plot_example_3d(x_sat_orbdyn_stm:np.ndarray, x_obs_multiple:np.ndarray):
    """ Plots a sphere, site position and satellite trajectory.

    Args:
        x_sat_orbdyn_stm (np.array): satellite trajectory array.
        x_obs_multiple (np.array): observer positions.
    """
    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(111, projection='3d')

    # Dimension fix
    if len(x_obs_multiple.shape) == 2:
        x_obs_multiple = np.expand_dims(x_obs_multiple)
    
    plot_sphere(ax1, d=R_EQ, n=40)

    s = []      # Scatter instances
    l = []      # Legends

    for i in range(x_obs_multiple.shape[2]):
        st = ax1.scatter(x_obs_multiple[0,:,0], x_obs_multiple[1,:,i], x_obs_multiple[2,:,i], marker='o')
        s.append(st)
        l.append(f"Observer {i}")

    s4 = ax1.scatter(x_sat_orbdyn_stm[0,:], x_sat_orbdyn_stm[1,:], x_sat_orbdyn_stm[2,:], marker='.', c='k',s=1)
    s.append(s4)
    l.append('Satellite')

    ax1.title.set_text('Scenario example')
    ax1.legend((s), (l), loc=2)

    return fig

def plot_range_range_rate(x_sat_orbdyn_stm:np.ndarray, x_obs_multiple:np.ndarray, t_sec: np.array):
    """ Plots range and range relative to the station

    Args:
        x_sat_orbdyn_stm (np.ndarray): satellite trajectory array.
        x_obs_multiple (np.ndarray): observer positions.
        t_sec (np.ndarray): array of timesteps.
    """

    if len(x_obs_multiple.shape) == 2:
        x_obs_multiple = np.expand_dims(x_obs_multiple)

    fig = plt.figure(figsize=(14,14))

    n_obs = x_obs_multiple.shape[2]

    for i in range(n_obs):
        r, rr = range_range_rate(x_sat_orbdyn_stm, x_obs_multiple[:,:,i])

        ax1 = fig.add_subplot(n_obs, 2, i*2+1)
        ax1.plot(t_sec, r)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Range (m)')
        ax1.grid(':')
        ax1.title.set_text('Station 1 - Range')

        ax2 = fig.add_subplot(n_obs, 2, i*2+2)
        ax2.plot(t_sec, rr)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Range rate (m/s)')
        ax2.grid(':')
        ax2.title.set_text('Station 1 - Range Rate')

    fig.subplots_adjust(hspace=0.25)

    return fig

def plot_pos_vel_norms(x_sat:np.ndarray, t_sec: np.array):
    """ Plots range and range relative to the station

    Args:
        x_sat_orbdyn_stm (np.ndarray): satellite trajectory array.
        x_obs_multiple (np.ndarray): observer positions.
        t_sec (np.ndarray): array of timesteps.
    """

    r = np.linalg.norm(x_sat[0:3,], axis=0)     # Norm of the position
    v = np.linalg.norm(x_sat[3:6,], axis=0)     # Norm of the velocity

    fig = plt.figure(figsize=(14,7))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t_sec, r)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Satellite position norm (m)')
    ax1.grid(':')
    ax1.title.set_text('Position Norm')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t_sec, v)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Satellite velocity norm (m/s)')
    ax2.grid(':')
    ax2.title.set_text('Velocity Norm')

    fig.subplots_adjust(hspace=0.25)

    return fig

def plot_batch_results(
        x_sat_orbdyn_stm:np.ndarray, 
        x_0r:np.ndarray, 
        x_br:np.ndarray, 
        x_berr:np.ndarray
    ):
    """ Plot relevant converged batch results.

    Args:
        x_sat_orbdyn_stm  (np.ndarray): True satellite position.
        x_0r (np.ndarray): array of random initial sampled positions.
        x_br (np.ndarray): array of batch estimates of initial positions.
        x_berr (np.ndarray): array of errors relative to x_0
    """

    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(111, projection='3d')

    # Groundtruth
    ax1.scatter(x_sat_orbdyn_stm[0,0], x_sat_orbdyn_stm[1,0], x_sat_orbdyn_stm[2,0], s=10, marker='x', c = 'r')

    x_berr_norm = np.linalg.norm(x_berr, axis=0)
    #norm_mask = x_berr_norm < 1000

    traj = ax1.plot(x_sat_orbdyn_stm[0,:2], x_sat_orbdyn_stm[1,:2], x_sat_orbdyn_stm[2,:2], c='r')

    # Batch results
    for i in range(x_0r.shape[1]):
        if x_berr_norm[i] < 1000:
            s1 = ax1.scatter(x_0r[0, i], x_0r[1, i], x_0r[2, i], c='b', s=1, marker='x')
            s2 = ax1.scatter(x_br[0, i], x_br[1, i], x_br[2, i], c='g', s=1)

    ax1.legend((traj, s1, s2), ('Groundtruth trajectory', 'Pre-batch positions', 'Post-batch positions'))

    return fig