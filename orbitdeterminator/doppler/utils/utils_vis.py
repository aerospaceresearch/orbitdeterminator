import os
import numpy as np
import matplotlib.pyplot as plt

from orbitdeterminator.doppler.utils.constants import *
from orbitdeterminator.doppler.utils.utils import *

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
        # TODO: Check first argument
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
    Returns:
        fig
    """

    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(111, projection='3d')

    # Groundtruth
    ax1.scatter(x_sat_orbdyn_stm[0,0], x_sat_orbdyn_stm[1,0], x_sat_orbdyn_stm[2,0], s=10, marker='x', c = 'r')

    x_berr_norm = np.linalg.norm(x_berr, axis=0)
    #norm_mask = x_berr_norm < 1000

    traj = ax1.plot(x_sat_orbdyn_stm[0,:2], x_sat_orbdyn_stm[1,:2], x_sat_orbdyn_stm[2,:2], c='r')
    traj_proxy = ax1.scatter(x_sat_orbdyn_stm[0,0], x_sat_orbdyn_stm[1,0], x_sat_orbdyn_stm[2,0], c='r', s=1)

    # Batch results
    for i in range(x_0r.shape[1]):
        if x_berr_norm[i] < 1000:
            s1 = ax1.scatter(x_0r[0, i], x_0r[1, i], x_0r[2, i], c='b', s=1, marker='x')
            s2 = ax1.scatter(x_br[0, i], x_br[1, i], x_br[2, i], c='g', s=1)

    s1_proxy = ax1.scatter(x_0r[0, 0], x_0r[1, 0], x_0r[2, 0], c='b', s=1, marker='x')
    s2_proxy = ax1.scatter(x_br[0, 1], x_br[1, 1], x_br[2, 1], c='g', s=1)

    ax1.legend((traj_proxy, s1_proxy, s2_proxy), ('Groundtruth trajectory', 'Pre-batch positions', 'Post-batch positions'))

    return fig

def plot_tdoa(tdoa:np.ndarray, tof:np.ndarray, t_sec:np.ndarray):
    """ Plot TDoA measurements.

    Args:
        tdoa (np.ndarray): time differential of arrival array (n_obs, n).
        tof (np.ndarray): time of flight array (n_obs, n).
        t_sec (np.ndarray): time array, seconds (n,).
    Returns:
        fig ():
    """
    fig = plt.figure(figsize=(14,7))
    fig.suptitle("Reference station time of flight (ToF) and time differential of arrival (TDoA) for other stations")
    # Reference station time of flight
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t_sec, tof[0,:])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Time of flight (s)')
    ax.grid(':')
    ax.title.set_text(f"Station 0 ToF")

    # Time differential of arrival for the rest of three stations
    for i in range(tdoa.shape[0]-1):
        ax = fig.add_subplot(2, 2, i+2)
        ax.plot(t_sec, tdoa[i+1,:])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Time differential (s)')
        ax.grid(':')
        ax.title.set_text(f"Station {i+1}-0 TDoA")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    fig.subplots_adjust(hspace=0.35)

    return fig

def plot_tdoa_results(p_sat:np.ndarray, x_obs:np.ndarray, x_sat:np.ndarray):
    """ Plot results of TDoA multilateration.

    Args:
        p_sat (np.ndarray): multilaterated satellite position (3, n).
        x_obs (np.ndarray): observer positions (6, n, n_obs).
        x_sat (np.ndarray): groundtruth satellite position (6, n).
    Returns:
        fig ():
    """ 
    x_obs_mean = np.mean(x_obs,axis=2)

    txtp, txtn = 1.002, 0.998    # Temporary variables - text location

    fig = plt.figure(figsize=(14,14))

    ax = fig.add_subplot(111, projection='3d')
    ax.title.set_text("TDoA Example")
    #plot_sphere(ax, d=R_EQ, n=40)

    # Observer
    obs = ax.scatter(x_obs[0,:,:], x_obs[1,:,:], x_obs[2,:,:], marker='o', s=1)
    for j in range(x_obs.shape[2]):
        ax.text(x_obs[0,0,j]*txtp, x_obs[1,0,j]*txtp, x_obs[2,0,j]*txtp, f"Observer {j}",c='b')
        ax.scatter(x_obs[0,:,:], x_obs[1,:,:], x_obs[2,:,:], marker='o', s=1)

    # Mean observer position
    ax.scatter(x_obs_mean[0, :], x_obs_mean[1, :], x_obs_mean[2, :], marker='.', s=1, alpha=0.1)
    ax.text(x_obs_mean[0, 0]*txtn, x_obs_mean[1, 0]*txtn, x_obs_mean[2, 0]*txtn, f"Observer (mean)")

    # Satellite
    sat = ax.scatter(x_sat[0,:], x_sat[1,:], x_sat[2,:], s=1)
    sat_0 = ax.scatter(x_sat[0,0], x_sat[1,0], x_sat[2,0], marker='x')
    ax.text(x_sat[0,0]*txtp, x_sat[1,0]*txtp, x_sat[2,0]*txtp, "Satellite")

    # Result trajectory
    res = ax.scatter(p_sat[0,:], p_sat[1,:], p_sat[2,:],alpha=0.1)

    ax.legend([res, sat, sat_0, obs],["Result Trajectory", "Groundtruth", "Start", "Observers"])

    return fig

def plot_tdoa_errors(p_sat, x_sat):
    """ Plots TDoA multilateration errors compared to groundtruth trajectory.

    Args:
        p_sat (np.ndarray): multilaterated satellite position (3, n).
        x_sat (np.ndarray): groundtruth satellite position (6, n).
    Returns:
        fig ():
    """
    tdoa_error = x_sat[0:3,:] - p_sat[0:3,:]

    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    ax.grid(':')
    ax.title.set_text("TDoA Multilateration Error, 4 stations, LEO Pass")
    xx = ax.plot(tdoa_error[0,:], linewidth=1)
    yy = ax.plot(tdoa_error[1,:], linewidth=1)
    zz = ax.plot(tdoa_error[2,:], linewidth=1)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Error (m)")

    ax.legend(["x","y","z"],loc=0)

    return fig

def save_images(x_sat, x_obs, t_sec=None, prefix="", path=""):
    """ Auxiliary function to save the images.

    Args:
        x_sat (np.ndarray): satellite state vectors (6,n).
        x_obs (np.ndarray): observer state vectors (6,n,n_ons).
        t_sec (np.ndarray): time array (n,).
        prefix (str): filename prefix.
        path (str): save path.
    Returns:
        None
    """

    fig_1 = plot_example_3d(x_sat, x_obs)
    fig_1.savefig(os.path.join(path, f"{prefix}_scenario"))

    fig_2 = plot_range_range_rate(x_sat, x_obs, t_sec)
    fig_2.savefig(os.path.join(path, f"{prefix}_range_range_rate"))

def save_images_batch_results(x_sat, x_0r, x_br, x_berr, prefix="", path=""):
    """ Auxiliary function to save the batch result images.

    Args:
        x_sat (np.ndarray): satellite state vectors (6,n). 
        x_0r  (np.ndarray): vector of pre-batch initial positions (6,n).
        x_br  (np.ndarray): vector if post-batch estimated initial positions (6,n).
        x_berr(np.ndarray): vector of errors (6,n).
        prefix (str): filename prefix.
        path (str): save path.
    Returns:
        None
    """

    fig_3 = plot_batch_results(x_sat, x_0r, x_br, x_berr)
    fig_3.savefig(os.path.join(path, f"{prefix}_range_range_rate"))