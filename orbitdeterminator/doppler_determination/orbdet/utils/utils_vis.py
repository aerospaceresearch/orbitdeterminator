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

    #if len(x_obs_multiple.shape) == 2:
    #    x_obs_multiple = np.expand_dims(x_obs_multiple)

    plot_sphere(ax1, d=R_EQ, n=40)
    s1 = ax1.scatter(x_obs_multiple[0,0,0], x_obs_multiple[1,0,0], x_obs_multiple[2,0,0], marker='o', c='r')
    s2 = ax1.scatter(x_obs_multiple[0,0,1], x_obs_multiple[1,0,1], x_obs_multiple[2,0,1], marker='o', c='g')
    s3 = ax1.scatter(x_obs_multiple[0,0,2], x_obs_multiple[1,0,2], x_obs_multiple[2,0,2], marker='o', c='b')
    s4 = ax1.scatter(x_sat_orbdyn_stm[0,:], x_sat_orbdyn_stm[1,:], x_sat_orbdyn_stm[2,:], marker='.', c='k',s=1)

    ax1.title.set_text('Scenario example')
    ax1.legend((s1,s2,s3,s4), ('Station 1', 'Station 2', 'Station 3', 'Satellite'), loc=2)

    return fig

def plot_range_range_rate(x_sat_orbdyn_stm:np.ndarray, x_obs_multiple:np.ndarray, t_sec):
    """ Plots range and range relative to the station
    """

    r_1, rr_1 = range_range_rate(x_sat_orbdyn_stm, x_obs_multiple[:,:,0])
    r_2, rr_2 = range_range_rate(x_sat_orbdyn_stm, x_obs_multiple[:,:,1])
    r_3, rr_3 = range_range_rate(x_sat_orbdyn_stm, x_obs_multiple[:,:,2])

    fig = plt.figure(figsize=(14,14))

    ax1 = fig.add_subplot(321)
    ax1.plot(t_sec, r_1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Range (m)')
    ax1.grid(':')
    ax1.title.set_text('Station 1 - Range')
    ax2 = fig.add_subplot(322)
    ax2.plot(t_sec, rr_1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Range rate (m/s)')
    ax2.grid(':')
    ax2.title.set_text('Station 1 - Range Rate')

    ax3 = fig.add_subplot(323)
    ax3.plot(t_sec, r_2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Range (m)')
    ax3.grid(':')
    ax3.title.set_text('Station 2 - Range')
    ax4 = fig.add_subplot(324)
    ax4.plot(t_sec, rr_2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Range rate (m/s)')
    ax4.grid(':')
    ax4.title.set_text('Station 2 - Range Rate')

    ax5 = fig.add_subplot(325)
    ax5.plot(t_sec, r_3)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Range (m)')
    ax5.grid(':')
    ax5.title.set_text('Station 3 - Range')
    ax6 = fig.add_subplot(326)
    ax6.plot(t_sec, rr_3)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Range rate (m/s)')
    ax6.grid(":")
    ax6.title.set_text('Station 3 - Range Rate')

    fig.subplots_adjust(hspace=0.25)

    return fig

def plot_batch_results(x_0, x_0r, x_br, x_berr, x_sat_orbdyn_stm, n_samples):
    """ Plot relevant converged batch results.
    """
    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(111, projection='3d')

    # Groundtruth
    ax1.scatter(x_0[0], x_0[1], x_0[2], s=10, marker='x', c = 'r')

    x_berr_norm = np.linalg.norm(x_berr, axis=0)
    norm_mask = x_berr_norm < 1000

    s1 = ax1.scatter(x_0r[0, 0], x_0r[1, 0], x_0r[2, 0], c='b', s=1, marker='x')
    s2 = ax1.scatter(x_br[0, 0], x_br[1, 0], x_br[2, 0], c='g', s=1)
    traj = ax1.plot(x_sat_orbdyn_stm[0,:2], x_sat_orbdyn_stm[1,:2], x_sat_orbdyn_stm[2,:2], c='r')

    # Batch results
    for i in range(n_samples):
        if x_berr_norm[i] < 1000:
            ax1.scatter(x_0r[0, i], x_0r[1, i], x_0r[2, i], c='b', s=1, marker='x')
            ax1.scatter(x_br[0, i], x_br[1, i], x_br[2, i], c='g', s=1)

    ax1.legend((traj, s1, s2), ('Groundtruth trajectory', 'Pre-batch positions', 'Post-batch positions'))

    return fig