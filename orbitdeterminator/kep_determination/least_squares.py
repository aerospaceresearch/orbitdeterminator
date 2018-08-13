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
from scipy.optimize import minimize
from scipy.optimize import least_squares
from orbitdeterminator.kep_determination.ellipse_fit import determine_kep, __read_file
from orbitdeterminator.kep_determination.gauss_method import *
from scipy.optimize import least_squares

# compute residuals vector, with Earth's grav parameter as to-be-fitted variable
def res_vec(x, my_data):

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
    iod_object_data = load_iod_data(filename)

    #load data of listed observatories (longitude, latitude, elevation)
    sat_observatories_data = load_sat_observatories_data('sat_tracking_observatories.txt')

    #get preliminary orbit using Gauss method
    #q0 : a, e, taup, I, W, w, T
    q0 = np.array(gauss_method_sat(filename, bodyname, obs_arr, r2_root_ind_vec=r2_root_ind_vec, refiters=gaussiters, plot=False))
    x0 = q0[0:6]
    x0[3:6] = np.deg2rad(x0[3:6])

    # if obs_arr_ls was not specified, then read whole data set:
    if obs_arr_ls is None:
        obs_arr_ls = np.array(range(1, len(iod_object_data)+1))

    rov = radec_obs_vec_sat(obs_arr_ls, iod_object_data)
    rv0 = radec_res_vec_rov_sat(x0, obs_arr_ls, iod_object_data, sat_observatories_data, rov)
    Q0 = np.linalg.norm(rv0, ord=2)/len(rv0)

    Q_ls = least_squares(radec_res_vec_rov_sat, x0, args=(obs_arr_ls, iod_object_data, sat_observatories_data, rov), method='lm', xtol=1e-13)

    print('\n*** ORBIT DETERMINATION: LEAST-SQUARES FIT ***')

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
    mpc_observatories_data = load_mpc_observatories_data('mpc_observatories.txt')

    #x0 : a, e, taup, I, W, w
    x0 = np.array(gauss_method_mpc(filename, bodyname, obs_arr, r2_root_ind_vec=r2_root_ind_vec, refiters=gaussiters, plot=False))

    x0[3:6] = np.deg2rad(x0[3:6])

    # if obs_arr_ls was not specified, then read whole data set:
    if obs_arr_ls is None:
        obs_arr_ls = np.array(range(1, len(mpc_object_data)+1))

    rov = radec_obs_vec_mpc(obs_arr_ls, mpc_object_data)

    rv0 = radec_res_vec_rov_mpc(x0, obs_arr_ls, mpc_object_data, mpc_observatories_data, rov)
    Q0 = np.linalg.norm(rv0, ord=2)/len(rv0)

    print('\n*** ORBIT DETERMINATION: LEAST-SQUARES FIT ***')

    Q_ls = least_squares(radec_res_vec_rov_mpc, x0, args=(obs_arr_ls, mpc_object_data, mpc_observatories_data, rov), method='lm')

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
    mu_Earth = 398600.435436E9 # m^3/seg^2
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
    n_ = meanmotion(mu_Earth,a_) #mean motion
    taup_ = data[0,0]-M_/n_ #time of pericenter passage

    # this is the vector of initial guess of orbital elements:
    x0 = np.array((a_, e_, taup_, omega_, I_, Omega_, mu_Earth))

    print('Orbital elements, initial guess:')
    print('Semi-major axis (a):                 ',a_,'m')
    print('Eccentricity (e):                    ',e_)
    print('Time of pericenter passage (tau):    ',taup_,'sec')
    print('Argument of pericenter (omega):      ',np.rad2deg(omega_),'deg')
    print('Inclination (I):                     ',np.rad2deg(I_),'deg')
    print('Longitude of Ascending Node (Omega): ',np.rad2deg(Omega_),'deg')
    print('Earth\'s G*mass                     : ',mu_Earth,'m^3 s^-2\n')

    #the arithmetic mean will be used as the reference epoch for the elements
    t_mean = np.mean(data[:,0])

    # minimize cost function QQ, using initial guess x0
    #Q_mini = minimize(QQ,x0,method='nelder-mead',options={'maxiter':100, 'disp': True})
    #Q_ls = least_squares(res_vec, x0, args=(data[0:2000,:], mu_Earth), method='lm')
    #Q_ls = least_squares(res_vec, x0, args=(data, mu_Earth), method='lm')
    Q_ls = least_squares(res_vec, x0, args=(data,), method='lm')
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
    print('True anomaly at epoch (f0):          ', np.rad2deg(time2truean(Q_ls.x[0], Q_ls.x[1], mu_Earth , t_mean, Q_ls.x[2])), 'deg')
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
    plt.plot( data[:,0], kep_r_(x0[0], x0[1], time2truean(x0[0], x0[1], mu_Earth, data[:,0], x0[2])), color="green", label='initial fit')
    plt.plot( data[:,0], kep_r_(Q_ls.x[0], Q_ls.x[1], time2truean(Q_ls.x[0], Q_ls.x[1], mu_Earth, data[:,0], Q_ls.x[2])), color="orange", label='LS fit')
    plt.xlabel('time')
    plt.ylabel('range')
    plt.title('LS fit vs observations: range')
    plt.legend()
    plt.show()

