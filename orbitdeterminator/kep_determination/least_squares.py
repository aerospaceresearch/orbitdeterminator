"""Computes the least-squares optimal Keplerian elements for a sequence of
   cartesian position observations.
"""

# # DEVELOPMENT ROADMAP:
# # write function to compute range as a function of orbital elements: DONE
# # write function to compute true anomaly as a function of time-of-fly: DONE
# # the following transformation is needed: from time t, to mean anomaly M,
# # to eccentric anomaly E, to true anomaly f, i.e.:
# # t -> M=n*(t-taup) -> M=E-e*sin(E) (invert) ->
# # -> f = 2*atan(  sqrt((1+e)/(1-e))*tan(E/2)  ): DONE
# # write function which takes observed values and computes the difference wrt expected-to-be-observed values as a function of unknown orbital elements (to be fitted): DONE
# # compute Q as a function of unknown orbital elements (to be fitted): DONE
# # optimize Q -> return fitted orbital elements (requires an ansatz: take input from minimalistic Gibb's?)

# NOTES to self:
# matrix multiplication of numpy's 2-D arrays is done through `np.matmul`

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from kep_determination.ellipse_fit import determine_kep, __read_file
from kep_determination.gauss_method import *
import kep_determination.positional_observation_reporting as por

# compute residuals vector, with Earth's grav parameter as to-be-fitted variable
def res_vec(x, my_data,weights):
    rv = np.zeros((3*my_data.shape[0]))
    for i in range(0,my_data.shape[0]-1):
        # observed xyz values
        xyz_obs = my_data[i,1:4]
        # predicted )computed xyz values
        xyz_com = orbel2xyz(my_data[i,0], x[6], x[0], x[1], x[2], x[3], x[4], x[5])
        # observed minus computed residual:
        rv[3*i-3] = xyz_obs[0]-xyz_com[0]
        rv[3*i-2] = xyz_obs[1]-xyz_com[1]
        rv[3*i-1] = xyz_obs[2]-xyz_com[2]

        rv[3*i-3] = weights[i]*rv[3*i-3]
        rv[3*i-2] = weights[i]*rv[3*i-2]
        rv[3*i-1] = weights[i]*rv[3*i-1]
    return rv

def res_vec_1(x, my_data):
    rv = np.zeros((3*my_data.shape[0]))
    for i in range(0,my_data.shape[0]-1):
        # observed xyz values
        xyz_obs = my_data[i,1:4]
        # predicted )computed xyz values
        xyz_com = orbel2xyz(my_data[i,0], x[6], x[0], x[1], x[2], x[3], x[4], x[5])
        # observed minus computed residual:
        rv[3*i-3] = xyz_obs[0]-xyz_com[0]
        rv[3*i-2] = xyz_obs[1]-xyz_com[1]
        rv[3*i-1] = xyz_obs[2]-xyz_com[2]
    return rv

def get_weights(resid):
    """
    This function calculates the weights per (x,y,z) by using the inverse of the squared residuals divided by the total sum of the inverse of the squared residuals.
    """
    total = sum([abs(resid[i]) for i in range(len(resid))])
    fract = np.array([resid[i]/total for i in range(len(resid))])
    return fract

def radec_res_vec_rov_sat_w(x, inds, iod_object_data, sat_observatories_data, rov, weights):
    """Compute vector of observed minus computed (O-C) weighted residuals for ra/dec Earth-orbiting satellite observations
    with pre-computed observed radec values vector. Assumes ra/dec observed values vector
    is contained in rov, and they are stored as rov = [ra1, dec1, ra2, dec2, ...].

       Args:
           x (1x6 float array): set of orbital elements (a, e, taup, omega, I, Omega)
           inds (int array): line numbers of data in file
           iod_object_data (ndarray): observation data
           sat_observatories_data (ndarray): satellite tracking stations data
           rov (1xlen(inds) float-like array): vector of observed ra/dec values

       Returns:
           rv (1xlen(inds) array): vector of ra/dec weighted (O-C) residuals.
    """
    rv = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1],
                       minutes=1.0*iod_object_data['min'][indm1],
                       seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1], iod_object_data['month'][indm1], iod_object_data['day'][indm1]) + td )
        site_code = iod_object_data['station'][indm1]
        obsite = por.get_station_data(site_code, sat_observatories_data)
        # object position wrt to Earth
        xyz_obj = orbel2xyz(timeobs.jd, cts.GM_earth.to(uts.Unit('km3 / day2')).value, x[0], x[1], x[2], x[3], x[4], x[5])
        # observer position wrt to Earth
        xyz_oe = observerpos_sat(obsite['Latitude'], obsite['Longitude'], obsite['Elev'], timeobs)
        # object position wrt observer (unnormalized LOS vector)
        rho_vec = xyz_obj - xyz_oe
        # compute normalized LOS vector
        rho_vec_norm = np.linalg.norm(rho_vec, ord=2)
        rho_vec_unit = rho_vec/rho_vec_norm
        # compute RA, Dec
        cosd_cosa = rho_vec_unit[0]
        cosd_sina = rho_vec_unit[1]
        sind = rho_vec_unit[2]
        # make sure computed RA (ra_comp) is always within [0.0, 2.0*np.pi]
        ra_comp = np.mod(np.arctan2(cosd_sina, cosd_cosa), 2.0*np.pi)
        dec_comp = np.arcsin(sind)
        #compute angle difference, taking always the smallest difference
        diff_ra = angle_diff_rad(rov[2*i-2], ra_comp)
        diff_dec = angle_diff_rad(rov[2*i-1], dec_comp)
        # store O-C residual into vector (O-C = "Observed minus Computed")
        rv[2*i-2], rv[2*i-1] = diff_ra, diff_dec
        rv[2*i-2]= weights[i]*rv[2*i-2]
        rv[2*i-1]= weights[i]*rv[2*i-1]
    return rv

def radec_res_vec_rov_mpc_w(x, inds, mpc_object_data, mpc_observatories_data, rov, weights):
    """Compute vector of observed minus computed weighted (O-C) residuals for ra/dec
    MPC-formatted observations of minor planets (asteroids, comets, etc.), with
    pre-computed observed radec values vector. Assumes ra/dec observed values
    vector is contained in rov, and they are stored as
    rov = [ra1, dec1, ra2, dec2, ...].

       Args:
           x (1x6 float array): set of orbital elements (a, e, taup, omega, I, Omega)
           inds (int array): line numbers of data in file
           mpc_object_data (ndarray): observation data
           mpc_observatories_data (ndarray): MPC observatories data
           rov (1xlen(inds) float-like array): vector of observed ra/dec values

       Returns:
           rv (1xlen(inds) array): vector of ra/dec (O-C) residuals.
    """
    rv = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        timeobs = Time( datetime(mpc_object_data['yr'][indm1], mpc_object_data['month'][indm1], mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
        site_code = mpc_object_data['observatory'][indm1]
        obsite = get_observatory_data(site_code, mpc_observatories_data)
        # compute residuals
        radec_res = radec_residual_rov_mpc(x, timeobs, rov[2*i-2], rov[2*i-1], obsite['Long'], obsite['sin'], obsite['cos'])
        # assign residuals to ra/dec residuals vector
        rv[2*i-2], rv[2*i-1] = radec_res
        rv[2*i-2]= weights[i]*rv[2*i-2]
        rv[2*i-1]= weights[i]*rv[2*i-1]
    return rv

# evaluate cost function given a set of observations
def Q(x, my_data):
    Q0 = 0.0
    for i in range(0,my_data.shape[0]-1):
        # observed xyz values
        xyz_obs = my_data[i,1:4]
        # predicted (computed) xyz values
        xyz_com = orbel2xyz(my_data[i,0], x[6], x[0], x[1], x[2], x[3], x[4], x[5])
        # observed minus computed residual:
        xyz_res = xyz_obs-xyz_com
        #square residual, add to total cost function, divide by number of observations
        Q0 = Q0 + np.linalg.norm(xyz_res, ord=2)/my_data.shape[0]
    return Q0

# cost function of only one argument, x
# due to optimization of processing time, only the first 2,000 data points are used
# nevertheless, this is enough to improve the solution
def QQ(x):
    return Q(x, data)

def gauss_LS_sat(filename, bodyname, obs_arr, r2_root_ind_vec=None, obs_arr_ls=None, gaussiters=0, plot=True):
    """Earth satellites orbit determination high-level function from
    IOD-formatted ra/dec tracking data. IOD angle subformat 2 is assumed.
    Preliminary orbit determination via Gauss method is performed.
    Roots of 8-th order Gauss polynomial are computed using np.roots function.
    Note that if `r2_root_ind_vec` is not specified by the user, then the first
    positive root returned by np.roots is used by default.

       Args:
           filename (string): path to IOD-formatted observation data file
           bodyname (string): user-defined name of satellite
           obs_arr (int vector): line numbers in data file to be processed in Gauss preliminary orbit determination
           r2_root_ind_vec (1xlen(obs_arr) int array): indices of Gauss polynomial roots.
           obs_arr (int vector): line numbers in data file to be processed in least-squares fit
           gaussiters (int): number of refinement iterations to be performed
           plot (bool): if True, plots data.

       Returns:
           x (tuple): set of Keplerian orbital elements (a, e, taup, omega, I, omega, T)
    """
    # load IOD data for a given satellite
    iod_object_data = por.load_iod_data(filename)

    #load data of listed observatories (longitude, latitude, elevation)
    sat_observatories_data = por.load_sat_observatories_data('../station_observatory_data/sat_tracking_observatories.txt')

    #get preliminary orbit using Gauss method
    #q0 : a, e, taup, I, W, w, T
    q0 = np.array(gauss_method_sat(filename,
                                   obs_arr=obs_arr,
                                   bodyname=bodyname,
                                   r2_root_ind_vec=r2_root_ind_vec,
                                   refiters=gaussiters,
                                   plot=False))
    x0 = q0[0:6]
    x0[3:6] = np.deg2rad(x0[3:6])

    # if obs_arr_ls was not specified, then read whole data set:
    if obs_arr_ls is None:
        obs_arr_ls = np.array(range(1, len(iod_object_data["yr"])+1))

    rov = radec_obs_vec_sat(obs_arr_ls, iod_object_data)
    rv0 = radec_res_vec_rov_sat(x0, obs_arr_ls, iod_object_data, sat_observatories_data, rov)
    Q0 = np.linalg.norm(rv0, ord=2)/len(rv0)

    print('\n*** ORBIT DETERMINATION: LEAST-SQUARES FIT ***')

    #Q_ls = least_squares(radec_res_vec_rov_sat, x0, args=(obs_arr_ls, iod_object_data, sat_observatories_data, rov), method='lm', xtol=1e-13)
    if(chk=='2'):
             Q_ls = least_squares(radec_res_vec_rov_sat, x0,
                                  args=(obs_arr_ls, iod_object_data, sat_observatories_data, rov),
                                  method='lm',
                                  xtol=1e-13)
             residuals=Q_ls.fun
             #print("--")
             #print(residuals)
             #print("--")
             weights=get_weights(residuals)
             #print("--")
             #print(weights)
             #print("--")
             Q_ls = least_squares(radec_res_vec_rov_sat_w, x0,
                                  args=(obs_arr_ls, iod_object_data, sat_observatories_data, rov, weights),
                                  method='lm',
                                  xtol=1e-13)

    elif(chk=='1'):
             Q_ls = least_squares(radec_res_vec_rov_sat, x0,
                                  args=(obs_arr_ls, iod_object_data, sat_observatories_data, rov),
                                  method='lm',
                                  xtol=1e-13)

    else:
             print("Invalid input.Exiting...")
             sys.exit()

    print('\nINFO: scipy.optimize.least_squares exited with code', Q_ls.status)
    print(Q_ls.message,'\n')

    tv_star, rv_star = t_radec_res_vec_sat(Q_ls.x, obs_arr_ls, iod_object_data, sat_observatories_data, rov)
    Q_star = np.linalg.norm(rv_star, ord=2)/len(rv_star)

    print('Total residual evaluated at averaged Gauss solution: ', Q0)
    print('Total residual evaluated at least-squares solution: ', Q_star, '\n')

    print('Observational arc:')
    print('Number of observations: ', len(obs_arr_ls))
    print('First observation (UTC) : ', Time(tv_star[0], format='jd').iso)
    print('Last observation (UTC) : ', Time(tv_star[-1], format='jd').iso)

    n_num = meanmotion(mu_Earth, Q_ls.x[0])

    print('\nORBITAL ELEMENTS (EQUATORIAL): a, e, taup, omega, I, Omega, T')
    print('Semi-major axis (a):                 ', Q_ls.x[0], 'km')
    print('Eccentricity (e):                    ', Q_ls.x[1])
    print('Time of pericenter passage (tau):    ', Time(Q_ls.x[2], format='jd').iso, 'JDUTC')
    print('Argument of pericenter (omega):      ', np.rad2deg(Q_ls.x[3]), 'deg')
    print('Inclination (I):                     ', np.rad2deg(Q_ls.x[4]), 'deg')
    print('Longitude of Ascending Node (Omega): ', np.rad2deg(Q_ls.x[5]), 'deg')
    print('Orbital period (T):                  ', 2.0*np.pi/n_num/60.0, 'min')

    # PLOT
    if plot:
        ra_res_vec = np.rad2deg(rv_star[0::2])*(3600.0)
        dec_res_vec = np.rad2deg(rv_star[1::2])*(3600.0)
        t_plot = (tv_star-tv_star[0])*86400.0

        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].set_title('Gauss + LS fit residuals: RA, Dec')
        axarr[0].scatter(t_plot, ra_res_vec, label='delta RA (\")', marker='+')
        axarr[0].set_ylabel('RA (\")')
        axarr[1].scatter(t_plot, dec_res_vec, label='delta Dec (\")', marker='+')
        axarr[1].set_xlabel('time (UTC seconds since first obs)')
        axarr[1].set_ylabel('Dec (\")')
        plt.show()

        npoints = 500
        theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
        x_orb_vec = np.zeros((npoints,))
        y_orb_vec = np.zeros((npoints,))
        z_orb_vec = np.zeros((npoints,))

        for i in range(0,npoints):
            x_orb_vec[i], y_orb_vec[i], z_orb_vec[i] = xyz_frame2(Q_ls.x[0], Q_ls.x[1], theta_vec[i], Q_ls.x[3], Q_ls.x[4], Q_ls.x[5])

        ax = plt.axes(aspect='equal', projection='3d')

        # Earth-centered orbits: satellite orbit and geocenter
        ax.scatter3D(0.0, 0.0, 0.0, color='blue', label='Earth')
        ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'red', linewidth=0.5, label=bodyname+' orbit')
        plt.legend()
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
        ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5))
        ax.set_title('Satellite orbit (Gauss+LS): '+bodyname)
        plt.show()

    return Q_ls.x[0], Q_ls.x[1], Time(Q_ls.x[2], format='jd'), np.rad2deg(Q_ls.x[3]), np.rad2deg(Q_ls.x[4]), np.rad2deg(Q_ls.x[5]), 2.0*np.pi/n_num/60.0

def gauss_LS_mpc(filename, bodyname, obs_arr, r2_root_ind_vec=None, obs_arr_ls=None, gaussiters=0, plot=True):
    """Minor planets orbit determination high-level function from MPC-formatted
    ra/dec tracking data. Preliminary orbit determination via Gauss method is
    performed. Roots of 8-th order Gauss polynomial are computed using np.roots
    function. Note that if `r2_root_ind_vec` is not specified by the user, then
    the first positive root returned by np.roots is used by default.

       Args:
           filename (string): path to MPC-formatted observation data file
           bodyname (string): user-defined name of minor planet
           obs_arr (int vector): line numbers in data file to be processed in Gauss preliminary orbit determination
           r2_root_ind_vec (1xlen(obs_arr) int array): indices of Gauss polynomial roots.
           obs_arr (int vector): line numbers in data file to be processed in least-squares fit
           gaussiters (int): number of refinement iterations to be performed
           plot (bool): if True, plots data.

       Returns:
           x (tuple): set of Keplerian orbital elements (a, e, taup, omega, I, omega, T)
    """
    # load MPC data for a given NEA
    mpc_object_data = load_mpc_data(filename)

    #load MPC data of listed observatories (longitude, parallax constants C, S)
    mpc_observatories_data = load_mpc_observatories_data('../station_observatory_data/mpc_observatories.txt')

    #x0 : a, e, taup, I, W, w
    x0 = np.array(gauss_method_mpc(filename, bodyname, obs_arr,
                                   r2_root_ind_vec=r2_root_ind_vec,
                                   refiters=gaussiters,
                                   plot=False))

    x0[3:6] = np.deg2rad(x0[3:6])

    # if obs_arr_ls was not specified, then read whole data set:
    if obs_arr_ls is None:
        obs_arr_ls = np.array(range(1, len(mpc_object_data)+1))

    rov = radec_obs_vec_mpc(obs_arr_ls, mpc_object_data)

    rv0 = radec_res_vec_rov_mpc(x0, obs_arr_ls, mpc_object_data, mpc_observatories_data, rov)
    Q0 = np.linalg.norm(rv0, ord=2)/len(rv0)

    print('\n*** ORBIT DETERMINATION: LEAST-SQUARES FIT ***')

    #Q_ls = least_squares(radec_res_vec_rov_mpc, x0, args=(obs_arr_ls, mpc_object_data, mpc_observatories_data, rov), method='lm')
    if(chk=='2'):
             Q_ls = least_squares(radec_res_vec_rov_mpc, x0,
                                  args=(obs_arr_ls, mpc_object_data, mpc_observatories_data, rov),
                                  method='lm')
             residuals=Q_ls.fun
             #print("--")
             #print(residuals)
             #print("--")
             weights=get_weights(residuals)
             #print("--")
             #print(weights)
             #print("--")
             Q_ls = least_squares(radec_res_vec_rov_mpc_w, x0,
                                  args=(obs_arr_ls, mpc_object_data, mpc_observatories_data, rov, weights),
                                  method='lm')
    elif(chk=='1'):
             Q_ls = least_squares(radec_res_vec_rov_mpc, x0,
                                  args=(obs_arr_ls, mpc_object_data, mpc_observatories_data, rov),
                                  method='lm')

    else:
             print("Invalid input.Exiting...")
             sys.exit()


    print('\nINFO: scipy.optimize.least_squares exited with code ', Q_ls.status)
    print(Q_ls.message,'\n')

    tv_star, rv_star = t_radec_res_vec_mpc(Q_ls.x, obs_arr_ls, mpc_object_data, mpc_observatories_data)
    Q_star = np.linalg.norm(rv_star, ord=2)/len(rv_star)

    print('Total residual evaluated at averaged Gauss solution: ', Q0)
    print('Total residual evaluated at least-squares solution: ', Q_star)

    print('Observational arc:')
    print('Number of observations: ', len(obs_arr_ls))
    print('First observation (UTC) : ', Time(tv_star[0], format='jd').iso)
    print('Last observation (UTC) : ', Time(tv_star[-1], format='jd').iso)

    n_num = meanmotion(mu_Sun, Q_ls.x[0])

    print('\nORBITAL ELEMENTS (ECLIPTIC, MEAN J2000.0): a, e, taup, omega, I, Omega, T')
    print('Semi-major axis (a):                 ', Q_ls.x[0], 'au')
    print('Eccentricity (e):                    ', Q_ls.x[1])
    print('Time of pericenter passage (tau):    ', Time(Q_ls.x[2], format='jd').iso, 'JDTDB')
    print('Pericenter distance (q):             ', Q_ls.x[0]*(1.0-Q_ls.x[1]), 'au')
    print('Apocenter distance (Q):              ', Q_ls.x[0]*(1.0+Q_ls.x[1]), 'au')
    print('Argument of pericenter (omega):      ', np.rad2deg(Q_ls.x[3]), 'deg')
    print('Inclination (I):                     ', np.rad2deg(Q_ls.x[4]), 'deg')
    print('Longitude of Ascending Node (Omega): ', np.rad2deg(Q_ls.x[5]), 'deg')
    print('Orbital period (T):                  ', 2.0*np.pi/n_num, 'days')

    # PLOT
    if plot:
        ra_res_vec = np.rad2deg(rv_star[0::2])*(3600.0)
        dec_res_vec = np.rad2deg(rv_star[1::2])*(3600.0)

        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].set_title('Gauss + LS fit residuals: RA, Dec')
        axarr[0].scatter(tv_star-tv_star[0], ra_res_vec, label='delta RA (\")', marker='+')
        axarr[0].set_ylabel('RA (\")')
        axarr[1].scatter(tv_star-tv_star[0], dec_res_vec, label='delta Dec (\")', marker='+')
        axarr[1].set_xlabel('time (TDB days since first obs)')
        axarr[1].set_ylabel('Dec (\")')
        plt.show()

        npoints = 500 # number of points in orbit
        theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
        t_Ea_vec = np.linspace(tv_star[0], tv_star[-1], npoints)
        x_orb_vec = np.zeros((npoints,))
        y_orb_vec = np.zeros((npoints,))
        z_orb_vec = np.zeros((npoints,))
        x_Ea_orb_vec = np.zeros((npoints,))
        y_Ea_orb_vec = np.zeros((npoints,))
        z_Ea_orb_vec = np.zeros((npoints,))

        for i in range(0,npoints):
            x_orb_vec[i], y_orb_vec[i], z_orb_vec[i] = xyz_frame2(Q_ls.x[0], Q_ls.x[1], theta_vec[i], Q_ls.x[3], Q_ls.x[4], Q_ls.x[5])
            xyz_Ea_orb_vec_equat = earth_ephemeris(t_Ea_vec[i])/au
            xyz_Ea_orb_vec_eclip = np.matmul(rot_equat_to_eclip, xyz_Ea_orb_vec_equat)
            x_Ea_orb_vec[i], y_Ea_orb_vec[i], z_Ea_orb_vec[i] = xyz_Ea_orb_vec_eclip

        ax = plt.axes(aspect='equal', projection='3d')

        # Sun-centered orbits: Computed orbit and Earth's
        ax.scatter3D(0.0, 0.0, 0.0, color='yellow', label='Sun')
        ax.plot3D(x_Ea_orb_vec, y_Ea_orb_vec, z_Ea_orb_vec, color='blue', linewidth=0.5, label='Earth orbit')
        ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'red', linewidth=0.5, label=bodyname+' orbit')
        plt.legend()
        ax.set_xlabel('x (au)')
        ax.set_ylabel('y (au)')
        ax.set_zlabel('z (au)')
        xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
        ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
        ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) #, ncol=3)
        ax.set_title('Angles-only orbit determ. (Gauss+LS): '+bodyname)
        plt.show()

    return Q_ls.x[0], Q_ls.x[1], Time(Q_ls.x[2], format='jd'), np.rad2deg(Q_ls.x[3]), np.rad2deg(Q_ls.x[4]), np.rad2deg(Q_ls.x[5]), 2.0*np.pi/n_num

if __name__ == "__main__":
    # Earth's mass parameter in appropriate units:
    # mu_Earth = 398600.435436E9 # m^3/sec^2
    mu_Earth = 398600.435436 # km^3/sec^2
    #Earth's radius in appropriate units:
    # R_Earth =  6378136.3 #m
    #minimal acceptable altitude for satellites (150 km)??
    #maximal acceptable altitude for satellites (150 km)??

    #write file name of data:
    fname = '../orbit.csv'

    # load observational data:
    data = np.loadtxt(fname,skiprows=1,usecols=(0,1,2,3))
    # generate vector of initial guess of orbital elements:
    # values written below correspond to solution of ellipse_fit.py for the same file
    data0 = __read_file(fname)
    kep0, res0 = determine_kep(data0)

    a_ = kep0[0][0] # m
    e_ = kep0[1][0]
    I_ = np.deg2rad(kep0[2][0]) #deg
    omega_ = np.deg2rad(kep0[3][0]) #deg
    Omega_ = np.deg2rad(kep0[4][0]) #deg
    f_ = np.deg2rad(kep0[5][0]) #deg

    #estimate time of pericenter passage from true anomaly at epoch
    E_ = truean2eccan(e_, f_) #ecc. anomaly
    M_ = E_-e_*np.sin(E_) #mean anomaly
    n_ = meanmotion(mu_Earth*(10**9),a_) #mean motion
    taup_ = data[0,0]-M_/n_ #time of pericenter passage

    # this is the vector of initial guess of orbital elements:
    x0 = np.array((a_, e_, taup_, omega_, I_, Omega_, mu_Earth*(10**9)))

    print('Orbital elements, initial guess:')
    print('Semi-major axis (a):                 ',a_,'m')
    print('Eccentricity (e):                    ',e_)
    print('Time of pericenter passage (tau):    ',taup_,'sec')
    print('Argument of pericenter (omega):      ',np.rad2deg(omega_),'deg')
    print('Inclination (I):                     ',np.rad2deg(I_),'deg')
    print('Longitude of Ascending Node (Omega): ',np.rad2deg(Omega_),'deg')
    print('Earth\'s G*mass                     : ',mu_Earth*(10**9),'m^3 s^-2\n')

    #the arithmetic mean will be used as the reference epoch for the elements
    t_mean = np.mean(data[:,0])

    # minimize cost function QQ, using initial guess x0
    #Q_mini = minimize(QQ,x0,method='nelder-mead',options={'maxiter':100, 'disp': True})
    #Q_ls = least_squares(res_vec, x0, args=(data[0:2000,:], mu_Earth), method='lm')
    #Q_ls = least_squares(res_vec, x0, args=(data, mu_Earth), method='lm')
    print("What action do you want to perform?")
    print("1.Least squares.")
    print("2.Weighted Least squares.")
    chk=input()


    if(chk=='2'):
             Q_ls = least_squares(res_vec_1, x0, args=(data,), method='lm')
             residuals=Q_ls.fun
             #print("--")
             #print(residuals)
             #print("--")
             weights=get_weights(residuals)
             #print("--")
             #print(weights)
             #print("--")
             Q_ls = least_squares(res_vec, x0, args=(data,weights), method='lm')

    elif(chk=='1'):
             Q_ls = least_squares(res_vec_1, x0, args=(data,), method='lm')
    else:
             print("Invalid input.Exiting...")
             sys.exit()

    print('scipy.optimize.least_squares exited with code ', Q_ls.status)
    print(Q_ls.message,'\n')
    #display least-squares solution
    print('\nOrbital elements, least-squares solution:')
    print('Reference epoch (t0):                ', t_mean)
    print('Semi-major axis (a):                 ', Q_ls.x[0], 'm')
    print('Eccentricity (e):                    ', Q_ls.x[1])
    print('Time of pericenter passage (tau):    ', Q_ls.x[2], 'sec')
    print('Pericenter distance (q):             ', Q_ls.x[0]*(1.0-Q_ls.x[1]), 'm')
    print('Apocenter distance (Q):              ', Q_ls.x[0]*(1.0+Q_ls.x[1]), 'm')
    print('True anomaly at epoch (f0):          ', np.rad2deg(time2truean(Q_ls.x[0], Q_ls.x[1], mu_Earth*(10**9) , t_mean, Q_ls.x[2])), 'deg')
    print('Argument of pericenter (omega):      ', np.rad2deg(Q_ls.x[3]), 'deg')
    print('Inclination (I):                     ', np.rad2deg(Q_ls.x[4]), 'deg')
    print('Longitude of Ascending Node (Omega): ', np.rad2deg(Q_ls.x[5]), 'deg')
    print('Earth\'s G*mass                     : ',Q_ls.x[6],' m^3 s^-2\n')

    print('Total residual evaluated at initial guess: ', QQ(x0))
    print('Total residual evaluated at least-squares solution: ', QQ(Q_ls.x))
    print('Percentage improvement: ', (QQ(x0)-QQ(Q_ls.x))/QQ(x0)*100, ' %')

    # the observed range as a function of time will be used for plotting
    ranges_ = np.sqrt(data[:,1]**2+data[:,2]**2+data[:,3]**2)

    #generate plots:
    plt.scatter( data[:,0], ranges_ ,s=0.1, label='observed data')
    plt.plot( data[:,0], kep_r_(x0[0], x0[1], time2truean(x0[0], x0[1], mu_Earth*(10**9), data[:,0], x0[2])),
              color="green",
              label='initial fit')
    plt.plot( data[:,0], kep_r_(Q_ls.x[0], Q_ls.x[1], time2truean(Q_ls.x[0], Q_ls.x[1], mu_Earth*(10**9), data[:,0], Q_ls.x[2])),
              color="orange",
              label='LS fit')
    plt.xlabel('time')
    plt.ylabel('range')
    plt.title('LS fit vs observations: range')
    plt.legend()
    plt.show()
