# This file demonstrates the use of the Gauss method, followed by a least-squares fit,
# in order to compute the orbit of an Earth-orbiting satellite, from ra/dec tracking data

from gauss_method import gauss_LS_sat
from numpy import zeros

# body name
body_name_str = 'USA 74 (21799)'

# path of file of ra/dec IOD-formatted observations
# the example contains tracking data for satellite USA 74
body_fname_str = '../example_data/iod_data_af2.txt'

#lines of observations file to be used for orbit determination
obs_arr = [1, 3, 4, 6, 8] # LB observations of 21799 91 076C on 2018 Jul 22

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[4] = 1 # modify as necessary if adequate root of Gauss polynomial has to be selected

x = gauss_LS_sat(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, gaussiters=10, plot=True)