# This file demonstrates the use of the Gauss method, followed by a least-squares fit,
# in order to compute the orbit of an Earth-orbiting satellite, from ra/dec tracking data

from gauss_method import gauss_LS_sat
from numpy import zeros

# body name
body_name_str = 'ISS (25544)'

# path of file of ra/dec IOD-formatted observations
# the example contains tracking data for ISS (25544)
body_fname_str = '../example_data/SATOBS-ML-19200716.txt'
# body_fname_str = '../example_data/iss_radec_generated_horizons.txt'

#lines of observations file to be used for preliminary orbit determination via Gauss method
obs_arr = [2, 3, 4, 5] # ML observations of ISS on 2016 Jul 19 and 20
# obs_arr = [7, 10, 13] # simulated observations generated from HORIZONS for ISS on 2018 Aug 08

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[4] = 1 # modify as necessary if adequate root of Gauss polynomial has to be selected

gauss_LS_sat(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, gaussiters=10, plot=True)
