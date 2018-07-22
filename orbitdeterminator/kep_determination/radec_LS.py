from least_squares import xyz_frame_
from least_squares import orbel2xyz
import gauss_method as gm
from datetime import datetime, timedelta
from jplephem.spk import SPK
import numpy as np
from astropy.coordinates import Longitude, Angle, SkyCoord
from astropy import units as uts
from astropy import constants as cts
from astropy.time import Time

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
mpc_object_data = gm.load_mpc_data(body_fname_str)
# print('MPC observation data:\n', mpc_object_data[ inds ], '\n')

#load MPC data of listed observatories (longitude, parallax constants C, S) (~7,000 observations)
mpc_observatories_data = gm.load_mpc_observatories_data('mpc_observatories.txt')

#lines of observations file to be used for orbit determination
obs_arr = [2341,2352,2362,2369,2377,2386,2387]

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)
r2_root_ind_vec[4] = 1 # uncomment and modify if adequate root of Gauss polynomial has to be selected

#x0 : a, e, taup, I, W, w
x0 = np.array(gm.gauss_method_mpc(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=5, plot=False))
x0[0] = 1.458251585462893
x0[1] = 0.2229072630918923
x0[2] = 2452085.0842563203
# x0[3] = 10.82944790594134
x0[4] = 304.4109222194975
x0[5] = 178.6283758645153
print('x0 = ', x0)
x0[3:6] = np.deg2rad(x0[3:6])
print('x0 = ', x0)

indx = obs_arr[4]
indm1 = indx-1

# extract observations data
# obs_radec, obs_t, site_codes = get_observations_data(mpc_object_data, inds)
timeobs = Time( datetime(mpc_object_data['yr'][indm1], mpc_object_data['month'][indm1], mpc_object_data['day'][indm1]) + timedelta(days=mpc_object_data['utc'][indm1]) )
site_code = mpc_object_data['observatory'][indm1]
obs_t_ra_dec = SkyCoord(mpc_object_data['radec'][indm1], unit=(uts.hourangle, uts.deg), obstime=timeobs)
obsite = gm.get_observatory_data(site_code, mpc_observatories_data)[0]
radec_res = gm.radec_residual(x0, obs_t_ra_dec, spk_kernel, obsite['Long'], obsite['sin'], obsite['cos'])

#radec_residuals x, inds (array of indices), mpc_object_data, mpc_observatories_data, spk_kernel

print('timeobs = ', timeobs)
print('obs_t_ra_dec = ', obs_t_ra_dec)
print('obs_t_ra_dec.obstime = ', obs_t_ra_dec.obstime)
print('obs_t_ra_dec.ra.deg = ', obs_t_ra_dec.ra.deg)
print('obs_t_ra_dec.dec.deg = ', obs_t_ra_dec.dec.deg)
print('site_code = ', site_code)
print('obsite = ', obsite)
print('obsite[\'Long\'] = ', obsite['Long'])
print('obsite[\'sin\'] = ', obsite['sin'])
print('obsite[\'cos\'] = ', obsite['cos'])
print('radec_res = ', radec_res)

# print(' = ', )

