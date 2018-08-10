from gauss_method import gauss_LS_mpc
from numpy import zeros

# path of file of optical MPC-formatted observations
body_fname_str = '../example_data/mpc_eros_data.txt'

#body name
body_name_str = 'Eros'

#lines of observations file to be used for preliminary orbit determination using Gauss method
obs_arr = [1, 14, 15, 24, 32, 37, 68, 81, 122, 162, 184, 206, 223] #2016 observations

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[4] = 1 # modify as necessary if adequate root of Gauss polynomial has to be selected

x = gauss_LS_mpc(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, gaussiters=10, plot=True)

