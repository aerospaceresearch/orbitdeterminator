import time, argparse, os, json
import numpy as np

from astropy.time import Time   # Astropy 4.1rc1 is used

from orbitdeterminator.doppler.utils.utils import *
from orbitdeterminator.doppler.utils.utils_aux import *
from orbitdeterminator.doppler.utils.utils_vis import *

def plot_position_norm_tdoa_batch(p_sat:np.ndarray, x_sat:np.ndarray, time:np.ndarray):
    """ Plot comparison between TDoA satellite position norm results and batch filter results.
    """

    x_sat_pos_norm = np.linalg.norm(x_sat[0:3,:], axis=0)
    p_sat_norm = np.linalg.norm(p_sat, axis = 0)

    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Estimated satellite position vector norm ||x||")
    ax = fig.add_subplot(111)

    #ax.set_ylim([6.378e6, 7.5e6])
    hg = ax.scatter(time, p_sat_norm, s=0.5)
    ba = ax.scatter(time[idx_start:idx_end], x_sat_pos_norm, s=0.5)
    ax.set_xlabel("Time (GPS seconds)")
    ax.set_ylabel("Sat R norm (m)")
    ax.grid(':')
    ax.legend([hg, ba], ["TDoA results", "Batch results"])

    return fig

def plot_batch_results_final(x_sat_result:np.ndarray, x_obs:np.ndarray):
    """ Plot batch results from final evaluation simulation.
    """

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    font = {'size': 16}
    matplotlib.rc('font', **font)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))

    # Satellite
    ax.scatter(x_sat_result[0,:], x_sat_result[1,:], x_sat_result[2,:], s=0.5, c='k')
    res = ax.scatter(x_sat_result[0,0], x_sat_result[1,0], x_sat_result[2,0], c='k')   # Proxy
    
    for j in range(x_obs.shape[2]):
        ax.scatter(x_obs[0,idx_start:idx_end,:], x_obs[1,idx_start:idx_end,:], x_obs[2,idx_start:idx_end,:], marker='.', s=0.5, c='b')

    obs = ax.scatter(x_obs[0,0,:], x_obs[1,0,:], x_obs[2,0,:], c='b')
    o = ax.scatter(0,0,0,c='teal')

    plot_sphere(ax, d=R_EQ, n=40)

    ax.set_xlabel("x ECI (m)", fontsize=16, labelpad=15)
    ax.set_ylabel("y ECI (m)", fontsize=16, labelpad=15)
    ax.set_zlabel("z ECI (m)", fontsize=16, labelpad=15)

    ax.legend([res, obs, o],["Result Trajectory", "Observers", "Origin"], loc=2, bbox_to_anchor=(0.15,0.9))

    return fig

def plot_full_orbit_final(x_b:np.ndarray):
    """ Plot full revolution of the final batch estimate.
    """

    t_sec_rev = np.arange(5760)
    x_sat_rev = odeint(orbdyn_2body, x_b.squeeze(), t_sec_rev, args=(MU,)).T

    #####   Plot full orbit     #####

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    font = {'size': 16}
    matplotlib.rc('font', **font)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))

    ax.set_xlabel("x ECI (m)", fontsize=16, labelpad=15)
    ax.set_ylabel("y ECI (m)", fontsize=16, labelpad=15)
    ax.set_zlabel("z ECI (m)", fontsize=16, labelpad=15)
    
    ax.scatter(x_sat_rev[0,:], x_sat_rev[1,:], x_sat_rev[2,:], s=0.5, c='k')
    rev = ax.scatter(x_sat_rev[0,0], x_sat_rev[1,0], x_sat_rev[2,0], c='k')
    o = ax.scatter(0,0,0,c='teal')

    ax.legend([rev, o],["Result Trajectory Single Orbit", "Origin"],loc=2, bbox_to_anchor=(0.15,0.9))

    plot_sphere(ax, d=R_EQ, n=40)

    return fig

if __name__ == "__main__": 

    # Load data
    data_json, data = parse_json_data("data/data_1hz.txt")
    n_s, n_m = data['n_s'], data['n_m']
    
    # Conver variables
    data_time = data['gpstime_unix'][:,0]
    t_sec = data_time - data_time[0]
    
    obstime = Time(data_time, format='unix')
    datetime = obstime
    datetime.format='iso'

    #print(datetime)

    sidereal = obstime.sidereal_time('mean', data['station_pos'][1,:,0]*u.deg)
    x_obs = get_site_temp(data['station_pos'], obstime)

    data_range = data['range'].T
    data_range_rate = data_range[:,1:] - data_range[:,:-1]
    data_range_rate = np.concatenate([data_range_rate[:,0].reshape((5,1)), data_range_rate], axis=1)

    f_t = 1.375e8   # Hz
    C = 3e8 
    data_doppler = data['doppler'].T
    data_doppler_range_rate = -data_doppler / f_t * C

    # TODO: Time
    print(data_range.shape)
    print(data_doppler_range_rate.shape)

    # Simulate TDoA
    tdoa, tof = get_tdoa_simulated_r(data['range'].T)

    # Solve TDoA
    p_sat, tau = solve_tdoa(tdoa, x_obs)

    # Herrick-Gibbs
    w = 10  # window size (assuming 1 second intervals in time array)
    x_sat_hg = np.zeros((6, n_m-w))
    error_hg = [None] * int(n_m-w)   # Error array for Herrick-Gibbs Method

    # Perform Herrick_Gibbs
    for i in range(p_sat.shape[1]-2*w):
        idx = np.array([i, i+w, i+2*w])
        x_sat_hg[:,i], error_hg[i] = herrick_gibbs(p_sat[:, idx], t_sec[idx], angle_checks=True)

        if error_hg[i] is not None:
            print(f"Index {i}, Error {error_hg[i]}")

    # Batch Filter - First 
    idx_start = 10 # 250 #300
    idx_end = 718 # 550 #600

    x_0 = np.expand_dims(x_sat_hg[:,idx_start], axis=1)     # Initial estimate
    P = np.diag([1, 1, 1, 1, 1, 1])*1e10                    # Initial uncertainty
    R_rr = np.eye(n_s)*1e-3                                 # Measurement uncertainty
    z_rr = -data['doppler'].T                               # Measurements

    t_sec_temp = t_sec[idx_start:idx_end] - t_sec[idx_start]

    x_b, output = batch(
        np.copy(x_0),
        P, 
        R_rr, 
        z = data_range_rate[:,idx_start:idx_end],
        t = t_sec_temp,
        x_obs=x_obs[:,idx_start:idx_end,:], 
        f_obs=f_obs_range_rate, 
        tolerance=1e-8,
        max_iterations=250
    )

    print(output['num_it'])
    print(x_0.T, np.linalg.norm(x_0.T[0:3], axis=0))
    print(x_b.T, np.linalg.norm(x_b.T[0:3], axis=0))

    x_sat_result = odeint(orbdyn_2body, x_b.squeeze(), t_sec_temp, args=(MU,)).T

    # Saves
    output_json = {'x_sat': x_sat_result.tolist(), 'gpstime_unix': data_time[idx_start:idx_end].tolist()}

    with open('data/result.txt', 'w') as out_file:
        json.dump(output_json, out_file)

    ##### Plot intermediate results - 3D
    fig_1 = plot_tdoa_results(p_sat, x_obs, x_sat=None, angle=(35, -140))

    ##### Plot position norm
    fig_2 = plot_position_norm_tdoa_batch(p_sat, x_sat_result, data_time)

    #####   Plot batch results      
    fig_3 = plot_batch_results_final(x_sat_result, x_obs)
    
    #####   Plot full orbit
    fig_4 = plot_full_orbit_final(x_b)

    plt.show()
