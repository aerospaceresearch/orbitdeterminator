import gauss_method as gm

# path of file of optical MPC-formatted observations
filename = '../example_data/mpc_ceres_data.txt'

#body name
bodyname = 'Ceres'

#lines of observations file to be used for orbit determination
obs_arr = [7145,7146,7148,7152,7155,7156,7157,7158,7159,7164,7172,7178,7185,7190,7197,7201,7205,7213,7214,7218,7219,7221,7222,7227,7231,7240,7241,7242,7250]

###modify r2_root_ind_vec as necessary if adequate root of Gauss polynomial has to be selected
###if r2_root_ind_vec is not specified, then the first positive root will always be selected by default
# r2_root_ind_vec = zeros((len(obs_arr)-2,), dtype=int)
###select adequate index of Gauss polynomial root
# r2_root_ind_vec[4] = 1

# x = a_mean, e_mean, taup_mean, I_mean, W_mean, w_mean, T_mean
x = gm.gauss_method_mpc(filename, bodyname, obs_arr)
