import numpy as np
import gauss_method as gm

# path of file of optical MPC-formatted observations
body_fname_str = '../example_data/mpc_ceres_data.txt'

#body name
body_name_str = 'Ceres'

#lines of observations file to be used for orbit determination
obs_arr = [7145,7146,7148,7152,7155,7156,7157,7158,7159,7164,7172,7178,7185,7190,7197,7201,7205,7213,7214,7218,7219,7221,7222,7227,7231,7240,7241,7242,7250]

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[4] = 1 # uncomment and modify if adequate root of Gauss polynomial has to be selected

x_vec, y_vec, z_vec, x_Ea_vec, y_Ea_vec, z_Ea_vec, a_vec, e_vec, I_vec, W_vec, w_vec = gm.gauss_method_mpc(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=5)
