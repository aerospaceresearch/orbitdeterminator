import gauss_method as gm

# path of file of optical MPC-formatted observations
filename = '../example_data/mpc_eros_data.txt'

#body name
bodyname = 'Eros'

#lines of observations file to be used for orbit determination
obs_arr = [1, 14, 15, 24, 32, 37, 68, 81, 122, 162, 184, 206, 223] #2016 subset

###modify r2_root_ind_vec as necessary if adequate root of Gauss polynomial has to be selected
###if r2_root_ind_vec is not specified, then the first positive root will always be selected by default
# r2_root_ind_vec = zeros((len(obs_arr)-2,), dtype=int)
###select adequate index of Gauss polynomial root
# r2_root_ind_vec[4] = 1

# x = a_mean, e_mean, taup_mean, I_mean, W_mean, w_mean, T_mean
x = gm.gauss_method_mpc(filename, bodyname, obs_arr)
