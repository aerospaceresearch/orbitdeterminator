# TODO: evaluate Earth ephemeris only once for a given TDB instant
# this implies saving all UTC times and their TDB equivalencies

from least_squares import xyz_frame_, orbel2xyz, time2truean
import gauss_method as gm
from datetime import datetime, timedelta
from jplephem.spk import SPK
import numpy as np
from astropy.coordinates import Longitude, Angle, SkyCoord
from astropy import units as uts
from astropy import constants as cts
from astropy.time import Time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# compute auxiliary vector of observed ra,dec values
# inds = obs_arr
def radec_obs_vec(inds, mpc_object_data):
    rov = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        timeobs = Time( datetime(mpc_object_data['yr'][indm1], mpc_object_data['month'][indm1], mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
        obs_t_ra_dec = SkyCoord(mpc_object_data['radec'][indm1], unit=(uts.hourangle, uts.deg), obstime=timeobs)
        rov[2*i-2], rov[2*i-1] = obs_t_ra_dec.ra.rad, obs_t_ra_dec.dec.rad
    return rov

# compute residuals vector for ra/dec observations
# inds = obs_arr
def radec_res_vec(x, inds, mpc_object_data, mpc_observatories_data, spk_kernel):
    rv = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        # obs_radec, obs_t, site_codes = get_observations_data(mpc_object_data, inds)
        timeobs = Time( datetime(mpc_object_data['yr'][indm1], mpc_object_data['month'][indm1], mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
        site_code = mpc_object_data['observatory'][indm1]
        obs_t_ra_dec = SkyCoord(mpc_object_data['radec'][indm1], unit=(uts.hourangle, uts.deg), obstime=timeobs)
        obsite = gm.get_observatory_data(site_code, mpc_observatories_data)
        radec_res = gm.radec_residual(x, obs_t_ra_dec, spk_kernel, obsite['Long'], obsite['sin'], obsite['cos'])

        # print('timeobs = ', timeobs)
        # print('obs_t_ra_dec = ', obs_t_ra_dec)
        # print('obs_t_ra_dec.obstime = ', obs_t_ra_dec.obstime)
        # print('obs_t_ra_dec.obstime.jd = ', obs_t_ra_dec.obstime.jd)
        # print('obs_t_ra_dec.ra.deg = ', obs_t_ra_dec.ra.deg)
        # print('obs_t_ra_dec.dec.deg = ', obs_t_ra_dec.dec.deg)
        # print('site_code = ', site_code)
        # print('obsite = ', obsite)
        # print('obsite[\'Long\'] = ', obsite['Long'])
        # print('obsite[\'sin\'] = ', obsite['sin'])
        # print('obsite[\'cos\'] = ', obsite['cos'])
        # print('radec_res = ', radec_res)
        # observed minus computed residual:
        rv[2*i-2], rv[2*i-1] = radec_res
    return rv

# compute residuals vector for ra/dec observations with pre-computed observed radec values vector
# inds = obs_arr
def radec_res_vec_rov(x, inds, mpc_object_data, mpc_observatories_data, spk_kernel, rov):
    rv = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        # obs_radec, obs_t, site_codes = get_observations_data(mpc_object_data, inds)
        timeobs = Time( datetime(mpc_object_data['yr'][indm1], mpc_object_data['month'][indm1], mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
        site_code = mpc_object_data['observatory'][indm1]
        # obs_t_ra_dec = SkyCoord(mpc_object_data['radec'][indm1], unit=(uts.hourangle, uts.deg), obstime=timeobs)
        obsite = gm.get_observatory_data(site_code, mpc_observatories_data)
        # radec_res = gm.radec_residual(x, obs_t_ra_dec, spk_kernel, obsite['Long'], obsite['sin'], obsite['cos'])
        radec_res = gm.radec_residual_rov(x, timeobs, rov[2*i-2], rov[2*i-1], spk_kernel, obsite['Long'], obsite['sin'], obsite['cos'])

        # print('timeobs = ', timeobs)
        # print('obs_t_ra_dec = ', obs_t_ra_dec)
        # print('obs_t_ra_dec.obstime = ', obs_t_ra_dec.obstime)
        # print('obs_t_ra_dec.obstime.jd = ', obs_t_ra_dec.obstime.jd)
        # print('obs_t_ra_dec.ra.deg = ', obs_t_ra_dec.ra.deg)
        # print('obs_t_ra_dec.dec.deg = ', obs_t_ra_dec.dec.deg)
        # print('site_code = ', site_code)
        # print('obsite = ', obsite)
        # print('obsite[\'Long\'] = ', obsite['Long'])
        # print('obsite[\'sin\'] = ', obsite['sin'])
        # print('obsite[\'cos\'] = ', obsite['cos'])
        # print('radec_res = ', radec_res)
        # observed minus computed residual:
        rv[2*i-2], rv[2*i-1] = radec_res
    return rv

# compute residuals vector for ra/dec observations; return observation times and residual vector
# inds = obs_arr
def t_radec_res_vec(x, inds, mpc_object_data, mpc_observatories_data, spk_kernel):
    rv = np.zeros((2*len(inds)))
    tv = np.zeros((len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        # obs_radec, obs_t, site_codes = get_observations_data(mpc_object_data, inds)
        timeobs = Time( datetime(mpc_object_data['yr'][indm1], mpc_object_data['month'][indm1], mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
        site_code = mpc_object_data['observatory'][indm1]
        obs_t_ra_dec = SkyCoord(mpc_object_data['radec'][indm1], unit=(uts.hourangle, uts.deg), obstime=timeobs)
        obsite = gm.get_observatory_data(site_code, mpc_observatories_data)
        radec_res = gm.radec_residual(x, obs_t_ra_dec, spk_kernel, obsite['Long'], obsite['sin'], obsite['cos'])

        # print('timeobs = ', timeobs)
        # print('obs_t_ra_dec = ', obs_t_ra_dec)
        # print('obs_t_ra_dec.obstime = ', obs_t_ra_dec.obstime)
        # print('obs_t_ra_dec.obstime.jd = ', obs_t_ra_dec.obstime.jd)
        # print('obs_t_ra_dec.ra.deg = ', obs_t_ra_dec.ra.deg)
        # print('obs_t_ra_dec.dec.deg = ', obs_t_ra_dec.dec.deg)
        # print('site_code = ', site_code)
        # print('obsite = ', obsite)
        # print('obsite[\'Long\'] = ', obsite['Long'])
        # print('obsite[\'sin\'] = ', obsite['sin'])
        # print('obsite[\'cos\'] = ', obsite['cos'])
        # print('radec_res = ', radec_res)
        # observed minus computed residual:
        rv[2*i-2], rv[2*i-1] = radec_res
        tv[i] = timeobs.tdb.jd
    return tv, rv

# path of file of optical MPC-formatted observations
body_fname_str = '../example_data/mpc_eros_data.txt'

#body name
body_name_str = 'Eros'

# load JPL DE430 ephemeris SPK kernel, including TT-TDB difference
# 'de430t.bsp' may be downloaded from
# ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de430t.bsp
spk_kernel = SPK.open('de430t.bsp')
# print(spk_kernel)

# load MPC data for a given NEA
# mpc_object_data_unfiltered = gm.load_mpc_data(body_fname_str)
# # print('mpc_object_data_unfiltered = ', mpc_object_data)
# mpc_object_data = mpc_object_data_unfiltered[mpc_object_data_unfiltered['observatory'] != 'C51'][0]
# # print('mpc_object_data = ', mpc_object_data)
mpc_object_data = gm.load_mpc_data(body_fname_str)

# print('MPC observation data:\n', mpc_object_data[ inds ], '\n')

#load MPC data of listed observatories (longitude, parallax constants C, S) (~7,000 observations)
mpc_observatories_data = gm.load_mpc_observatories_data('mpc_observatories.txt')

#lines of observations file to be used for orbit determination
# obs_arr = [2341,2352,2362,2369,2377,2386,2387]
obs_arr = [7475, 7488, 7489, 7498, 7506, 7511, 7564, 7577, 7618, 7658, 7680, 7702, 7719] #2016 observations

# print('** = ', mpc_object_data)
# print('* = ', mpc_object_data_unfiltered)
# print('*** = ', mpc_object_data_unfiltered[['yr','month','day','utc']][obs_arr])
# print('*** = ', mpc_object_data_unfiltered[obs_arr])

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[4] = 1 # modify as necessary if adequate root of Gauss polynomial has to be selected

#x0 : a, e, taup, I, W, w
x0 = np.array(gm.gauss_method_mpc(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=5, plot=False))
# x0[0] = 1.458251585462893
# x0[1] = 0.2229072630918923
# x0[2] = 2452085.0842563203
# x0[3] = 178.6283758645153
# x0[4] = 10.82944790594134
# x0[5] = 304.4109222194975

# print('x0 = ', x0)
x0[3:6] = np.deg2rad(x0[3:6])
print('x0 = ', x0)

# obs_arr_ls = list(range(2341,2387))
obs_arr_ls1 = np.array(range(7475,7539))
obs_arr_ls2 = np.array(range(7562,7719))
obs_arr_ls = np.concatenate([obs_arr_ls1, obs_arr_ls2])
# obs_arr_ls = np.array(range(7475,7485))
# print('obs_arr_ls = ', obs_arr_ls)
print('obs_arr_ls[0] = ', obs_arr_ls[0])
print('obs_arr_ls[-1] = ', obs_arr_ls[-1])
nobs_ls = len(obs_arr_ls)
print('nobs_ls = ', nobs_ls)

rov = radec_obs_vec(obs_arr_ls, mpc_object_data)
# print('rov = ', rov)
# print('len(rov) = ', len(rov))

#         radec_res_vec_rov(x, inds, mpc_object_data, mpc_observatories_data, spk_kernel, rov)

rv0 = radec_res_vec(x0, obs_arr_ls, mpc_object_data, mpc_observatories_data, spk_kernel)
Q0 = np.linalg.norm(rv0, ord=2)/len(rv0)

# print('rv0 = ', rv0)
print('Q0 = ', Q0)

Q_ls = least_squares(radec_res_vec_rov, x0, args=(obs_arr_ls, mpc_object_data, mpc_observatories_data, spk_kernel, rov), method='lm')

print('scipy.optimize.least_squares exited with code ', Q_ls.status)
print(Q_ls.message,'\n')
print('Q_ls.x = ', Q_ls.x)

tv_star, rv_star = t_radec_res_vec(Q_ls.x, obs_arr_ls, mpc_object_data, mpc_observatories_data, spk_kernel)
Q_star = np.linalg.norm(rv_star, ord=2)/len(rv_star)
# print('rv* = ', rv_star)
print('Q* = ', Q_star)

print('Total residual evaluated at Gauss solution: ', Q0)
print('Total residual evaluated at least-squares solution: ', Q_star, '\n')
# print('Percentage improvement: ', (Q0-Q_star)/Q0*100, ' %')

print('\nOrbital elements, least-squares solution:')
# print('Reference epoch (t0):                ', t_mean)
print('Semi-major axis (a):                 ', Q_ls.x[0], 'au')
print('Eccentricity (e):                    ', Q_ls.x[1])
print('Time of pericenter passage (tau):    ', Time(Q_ls.x[2], format='jd').iso, 'JDTDB')
print('Pericenter distance (q):             ', Q_ls.x[0]*(1.0-Q_ls.x[1]), 'au')
print('Apocenter distance (Q):              ', Q_ls.x[0]*(1.0+Q_ls.x[1]), 'au')
# print('True anomaly at epoch (f0):          ', np.rad2deg(time2truean(Q_ls.x[0], Q_ls.x[1], gm.mu_Sun, t_mean, Q_ls.x[2])), 'deg')
print('Argument of pericenter (omega):      ', np.rad2deg(Q_ls.x[3]), 'deg')
print('Inclination (I):                     ', np.rad2deg(Q_ls.x[4]), 'deg')
print('Longitude of Ascending Node (Omega): ', np.rad2deg(Q_ls.x[5]), 'deg')

ra_res_vec = np.rad2deg(rv_star[0::2])*(3600.0)
dec_res_vec = np.rad2deg(rv_star[1::2])*(3600.0)

print('len(ra_res_vec) = ', len(ra_res_vec))
print('len(dec_res_vec) = ', len(dec_res_vec))
print('nobs_ls = ', nobs_ls)
print('len(tv_star) = ', len(tv_star))
# print('tv_star = ', tv_star)

# ax = plt.axes(aspect='equal', projection='3d')

# # Sun-centered orbits: Computed orbit and Earth's
# ax.scatter3D(0.0, 0.0, 0.0, color='yellow', label='Sun')
# ax.scatter3D(x_Ea_vec, y_Ea_vec, z_Ea_vec, color='blue', marker='.', label='Earth orbit')
# ax.plot3D(x_Ea_orb_vec, y_Ea_orb_vec, z_Ea_orb_vec, color='blue', linewidth=0.5)
# ax.scatter3D(x_vec, y_vec, z_vec, color='red', marker='+', label=body_name_str+' orbit')
# ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'red', linewidth=0.5)
# plt.legend()
# ax.set_xlabel('x (au)')
# ax.set_ylabel('y (au)')
# ax.set_zlabel('z (au)')
# xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
# ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
# ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
# ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
# ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) #, ncol=3)
# ax.set_title('Angles-only orbit determ. (Gauss): '+body_name_str)

# y_rad = 0.001

f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title('Gauss + LS fit residuals: RA, Dec')
axarr[0].scatter(tv_star, ra_res_vec, s=0.75, label='delta RA (\")')
# axarr[0].set_xlabel('time (JDTDB)')
axarr[0].set_ylabel('RA (\")')
# axarr[0].set_title('Sharing X axis')
axarr[1].scatter(tv_star, dec_res_vec, s=0.75, label='delta Dec (\")')
axarr[1].set_xlabel('time (JDTDB)')
axarr[1].set_ylabel('Dec (\")')
# # plt.xlim(4,5)
# # plt.ylim(-y_rad, y_rad)

plt.show()

# print(' = ', )

