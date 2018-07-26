# TODO: evaluate Earth ephemeris only once for a given TDB instant
# this implies saving all UTC times and their TDB equivalencies

from least_squares import xyz_frame_, orbel2xyz, meanmotion
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
def radec_obs_vec(inds, iod_object_data):
    rov = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1], minutes=1.0*iod_object_data['min'][indm1], seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1], iod_object_data['month'][indm1], iod_object_data['day'][indm1]) + td )
        raHHMMmmm  = iod_object_data['raHH' ][indm1] + (iod_object_data['raMM' ][indm1]+iod_object_data['rammm' ][indm1]/1000.0)/60.0
        decDDMMmmm = iod_object_data['decDD'][indm1] + (iod_object_data['decMM'][indm1]+iod_object_data['decmmm'][indm1]/1000.0)/60.0
        obs_t_ra_dec = SkyCoord(ra=raHHMMmmm, dec=decDDMMmmm, unit=(uts.hourangle, uts.deg), obstime=timeobs)
        rov[2*i-2], rov[2*i-1] = obs_t_ra_dec.ra.rad, obs_t_ra_dec.dec.rad
    return rov

# compute residuals vector for ra/dec observations with pre-computed observed radec values vector
# inds = obs_arr
def radec_res_vec_rov(x, inds, iod_object_data, sat_observatories_data, rov):
    rv = np.zeros((2*len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1], minutes=1.0*iod_object_data['min'][indm1], seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1], iod_object_data['month'][indm1], iod_object_data['day'][indm1]) + td )
        site_code = iod_object_data['station'][indm1]
        obsite = gm.get_station_data(site_code, sat_observatories_data)
        # object position wrt to Earth
        xyz_obj = orbel2xyz(timeobs.jd, cts.GM_earth.to(uts.Unit('km3 / day2')).value, x[0], x[1], x[2], x[3], x[4], x[5])
        # observer position wrt to Earth
        xyz_oe = gm.observerpos_sat(obsite['Latitude'], obsite['Longitude'], obsite['Elev'], timeobs)
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
        diff_ra = gm.angle_diff_rad(rov[2*i-2], ra_comp)
        diff_dec = gm.angle_diff_rad(rov[2*i-1], dec_comp)
        # compute O-C residual (O-C = "Observed minus Computed")
        rv[2*i-2], rv[2*i-1] = diff_ra, diff_dec
    return rv

# compute residuals vector for ra/dec observations; return observation times and residual vector
# inds = obs_arr
def t_radec_res_vec(x, inds, iod_object_data, sat_observatories_data, rov):
    rv = np.zeros((2*len(inds)))
    tv = np.zeros((len(inds)))
    for i in range(0,len(inds)):
        indm1 = inds[i]-1
        # extract observations data
        td = timedelta(hours=1.0*iod_object_data['hr'][indm1], minutes=1.0*iod_object_data['min'][indm1], seconds=(iod_object_data['sec'][indm1]+iod_object_data['msec'][indm1]/1000.0))
        timeobs = Time( datetime(iod_object_data['yr'][indm1], iod_object_data['month'][indm1], iod_object_data['day'][indm1]) + td )
        t_jd = timeobs.jd
        site_code = iod_object_data['station'][indm1]
        obsite = gm.get_station_data(site_code, sat_observatories_data)
        # object position wrt to Earth
        xyz_obj = orbel2xyz(t_jd, cts.GM_earth.to(uts.Unit('km3 / day2')).value, x[0], x[1], x[2], x[3], x[4], x[5])
        # observer position wrt to Earth
        xyz_oe = gm.observerpos_sat(obsite['Latitude'], obsite['Longitude'], obsite['Elev'], timeobs)
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
        diff_ra = gm.angle_diff_rad(rov[2*i-2], ra_comp)
        diff_dec = gm.angle_diff_rad(rov[2*i-1], dec_comp)
        # compute O-C residual (O-C = "Observed minus Computed")
        rv[2*i-2], rv[2*i-1] = diff_ra, diff_dec
        tv[i] = t_jd
    return tv, rv

# path of file of optical IOD-formatted observations
# the example contains tracking data for satellite USA 74
body_fname_str = '../example_data/iod_data_af2.txt'

#body name
body_name_str = 'USA 74 (21799)'

#lines of observations file to be used for orbit determination
obs_arr = [1, 3, 4, 6, 8] # LB observations of 21799 91 076C on 2018 Jul 22

# load IOD data for a given satellite
iod_object_data = gm.load_iod_data(body_fname_str)
# print('IOD observation data:\n', iod_object_data[ np.array(obs_arr)-1 ], '\n')

#load data of listed observatories (longitude, latitude, elevation)
sat_observatories_data = gm.load_sat_observatories_data('sat_tracking_observatories.txt')
# print('sat_observatories_data = ', sat_observatories_data)

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[4] = 1 # modify as necessary if adequate root of Gauss polynomial has to be selected

#get preliminary orbit using Gauss method
#q0 : a, e, taup, I, W, w, T
q0 = np.array(gm.gauss_method_sat(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=10, plot=False))
x0 = q0[0:6]
# x0[0] = 
# x0[1] = 
# x0[2] = 
# x0[3] = 
# x0[4] = 
# x0[5] = 
# x0[6] = 

# print('x0 = ', x0)
x0[3:6] = np.deg2rad(x0[3:6])
# print('x0 = ', x0)

obs_arr_ls = np.array(range(1, 8+1))
print('obs_arr_ls = ', obs_arr_ls)
# print('obs_arr_ls[0] = ', obs_arr_ls[0])
# print('obs_arr_ls[-1] = ', obs_arr_ls[-1])
nobs_ls = len(obs_arr_ls)
# print('nobs_ls = ', nobs_ls)

rov = radec_obs_vec(obs_arr_ls, iod_object_data)
print('rov = ', rov)
print('len(rov) = ', len(rov))

rv0 = radec_res_vec_rov(x0, obs_arr_ls, iod_object_data, sat_observatories_data, rov)
Q0 = np.linalg.norm(rv0, ord=2)/len(rv0)

print('rv0 = ', rv0)
print('Q0 = ', Q0)

Q_ls = least_squares(radec_res_vec_rov, x0, args=(obs_arr_ls, iod_object_data, sat_observatories_data, rov), method='lm')

print('INFO: scipy.optimize.least_squares exited with code', Q_ls.status)
print(Q_ls.message,'\n')
print('Q_ls.x = ', Q_ls.x)

tv_star, rv_star = t_radec_res_vec(Q_ls.x, obs_arr_ls, iod_object_data, sat_observatories_data, rov)
Q_star = np.linalg.norm(rv_star, ord=2)/len(rv_star)
print('rv* = ', rv_star)
print('Q* = ', Q_star)

print('Total residual evaluated at Gauss solution: ', Q0)
print('Total residual evaluated at least-squares solution: ', Q_star, '\n')
# # print('Percentage improvement: ', (Q0-Q_star)/Q0*100, ' %')

print('Observational arc:')
print('Number of observations: ', len(obs_arr_ls))
print('First observation (UTC) : ', Time(tv_star[0], format='jd').iso)
print('Last observation (UTC) : ', Time(tv_star[-1], format='jd').iso)

n_num = meanmotion(gm.mu_Earth, Q_ls.x[0])

print('\nOrbital elements, Gauss + least-squares solution:')
# print('Reference epoch (t0):                ', t_mean)
print('Semi-major axis (a):                 ', Q_ls.x[0], 'km')
print('Eccentricity (e):                    ', Q_ls.x[1])
print('Time of pericenter passage (tau):    ', Time(Q_ls.x[2], format='jd').iso, 'JDUTC')
print('Pericenter altitude (q):             ', Q_ls.x[0]*(1.0-Q_ls.x[1])-gm.Re, 'km')
print('Apocenter altitude (Q):              ', Q_ls.x[0]*(1.0+Q_ls.x[1])-gm.Re, 'km')
# print('True anomaly at epoch (f0):          ', np.rad2deg(time2truean(Q_ls.x[0], Q_ls.x[1], gm.mu_Sun, t_mean, Q_ls.x[2])), 'deg')
print('Argument of pericenter (omega):      ', np.rad2deg(Q_ls.x[3]), 'deg')
print('Inclination (I):                     ', np.rad2deg(Q_ls.x[4]), 'deg')
print('Longitude of Ascending Node (Omega): ', np.rad2deg(Q_ls.x[5]), 'deg')
print('Orbital period (T):                  ', 2.0*np.pi/n_num/60.0, 'min')

ra_res_vec = np.rad2deg(rv_star[0::2])*(3600.0)
dec_res_vec = np.rad2deg(rv_star[1::2])*(3600.0)

# print('len(ra_res_vec) = ', len(ra_res_vec))
# print('len(dec_res_vec) = ', len(dec_res_vec))
# print('nobs_ls = ', nobs_ls)
# print('len(tv_star) = ', len(tv_star))
# # print('tv_star = ', tv_star)

# y_rad = 0.001

f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title('Gauss + LS fit residuals: RA, Dec')
axarr[0].scatter(tv_star, ra_res_vec, s=0.75, label='delta RA (\")')
axarr[0].set_ylabel('RA (\")')
axarr[1].scatter(tv_star, dec_res_vec, s=0.75, label='delta Dec (\")')
axarr[1].set_xlabel('time (JDUTC)')
axarr[1].set_ylabel('Dec (\")')
# # plt.xlim(4,5)
# # plt.ylim(-y_rad, y_rad)
plt.show()

npoints = 1000
theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
x_orb_vec = np.zeros((npoints,))
y_orb_vec = np.zeros((npoints,))
z_orb_vec = np.zeros((npoints,))

for i in range(0,npoints):
    x_orb_vec[i], y_orb_vec[i], z_orb_vec[i] = xyz_frame_(Q_ls.x[0], Q_ls.x[1], theta_vec[i], Q_ls.x[3], Q_ls.x[4], Q_ls.x[5])

ax = plt.axes(aspect='equal', projection='3d')

# Earth-centered orbits: Computed orbit and Earth's
ax.scatter3D(0.0, 0.0, 0.0, color='blue', label='Earth')
# ax.scatter3D(x_vec, y_vec, z_vec, color='red', marker='+')
ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'red', linewidth=0.5, label=body_name_str+' orbit')
plt.legend()
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
ax.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) #, ncol=3)
ax.set_title('Satellite orbit (Gauss+LS): '+body_name_str)
plt.show()

# print(' = ', )

