from least_squares import gauss_LS_mpc

###path of file of optical MPC-formatted observations
filename = '../example_data/mpc_eros_data.txt'

###body name
bodyname = 'Eros'

###vector of line numbers in observations file to be used for preliminary orbit determination using Gauss method
obs_arr = [1, 14, 15, 24, 32, 37, 68, 81, 122, 162, 184, 206, 223] #2016 observations

###vector of line numbers in observations file to be used in least squares fitting
###if obs_arr_ls is not specified, then the whole data set is used by default
# obs_arr_ls = array(range(obs_arr[0], obs_arr[-1]+1))

###modify r2_root_ind_vec as necessary if adequate root of Gauss polynomial has to be selected
###if r2_root_ind_vec is not specified, then the first positive root will always be selected by default
# r2_root_ind_vec = zeros((len(obs_arr)-2,), dtype=int)
###select adequate index of Gauss polynomial root
# r2_root_ind_vec[4] = 1

x = gauss_LS_mpc(filename, bodyname, obs_arr, gaussiters=10, plot=True)
