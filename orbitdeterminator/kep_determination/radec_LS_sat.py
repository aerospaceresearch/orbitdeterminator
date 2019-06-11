# This file demonstrates the use of the Gauss method, followed by a least-squares fit,
# in order to compute the orbit of an Earth-orbiting satellite, from ra/dec tracking data

from least_squares import gauss_LS_sat

# body name
bodyname = 'USA 74 (21799)'

# path of file of ra/dec IOD-formatted observations
# the example contains tracking data for satellite USA 74
filename = '../example_data/iod_data_af2.txt'

#lines of observations file to be used for orbit determination
obs_arr = [1, 3, 4, 6, 8] # LB observations of 21799 91 076C on 2018 Jul 22

#the total number of observations used
# nobs = len(obs_arr)

###modify r2_root_ind_vec as necessary if adequate root of Gauss polynomial has to be selected
###if r2_root_ind_vec is not specified, then the first positive root will always be selected by default
# r2_root_ind_vec = zeros((len(obs_arr)-2,), dtype=int)
###select adequate index of Gauss polynomial root
# r2_root_ind_vec[4] = 1

x = gauss_LS_sat(filename, bodyname, obs_arr, gaussiters=10, plot=True)