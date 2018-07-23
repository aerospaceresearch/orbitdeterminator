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
        obsite = gm.get_observatory_data(site_code, mpc_observatories_data)[0]
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
print('obs_arr_ls = ', obs_arr_ls)

rv0 = radec_res_vec(x0, obs_arr_ls, mpc_object_data, mpc_observatories_data, spk_kernel)
Q0 = np.linalg.norm(rv0, ord=2)/len(rv0)

# print('rv0 = ', rv0)
print('Q0 = ', Q0)

Q_ls = least_squares(radec_res_vec, x0, args=(obs_arr_ls, mpc_object_data, mpc_observatories_data, spk_kernel), method='lm')

print('scipy.optimize.least_squares exited with code ', Q_ls.status)
print(Q_ls.message,'\n')
print('Q_ls.x = ', Q_ls.x)

rv_star = radec_res_vec(Q_ls.x, obs_arr_ls, mpc_object_data, mpc_observatories_data, spk_kernel)
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

# print(' = ', )

