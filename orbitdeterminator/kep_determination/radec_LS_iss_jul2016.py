# This file demonstrates the use of the Gauss method, followed by a least-squares fit,
# in order to compute the orbit of an Earth-orbiting satellite, from ra/dec tracking data

from least_squares import gauss_LS_sat

# body name
bodyname = 'ISS (25544)'

# path of file of ra/dec IOD-formatted observations
# the example contains tracking data for ISS (25544)
filename = '../example_data/SATOBS-ML-19200716.txt'
# filename = '../example_data/iss_radec_generated_horizons.txt'

#lines of observations file to be used for preliminary orbit determination via Gauss method
obs_arr = [2, 3, 4, 5] # ML observations of ISS on 2016 Jul 19 and 20
# obs_arr = [7, 10, 13] # simulated observations generated from HORIZONS for ISS on 2018 Aug 08

###vector of line numbers in observations file to be used in least squares fitting
###if obs_arr_ls is not specified, then the whole data set is used by default
# obs_arr_ls = array(range(obs_arr[0], obs_arr[-1]+1))

###modify r2_root_ind_vec as necessary if adequate root of Gauss polynomial has to be selected
###if r2_root_ind_vec is not specified, then the first positive root will always be selected by default
# r2_root_ind_vec = zeros((len(obs_arr)-2,), dtype=int)
###select adequate index of Gauss polynomial root
# r2_root_ind_vec[4] = 1

x = gauss_LS_sat(filename, bodyname, obs_arr, gaussiters=10, plot=True)
