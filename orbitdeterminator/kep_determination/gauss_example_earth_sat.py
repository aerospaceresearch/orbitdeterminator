import numpy as np
import gauss_method as gm

# path of file of optical IOD-formatted observations
body_fname_str = '../example_data/iod_data_af2.txt'

#body name
body_name_str = '43145'

#lines of observations file to be used for orbit determination
obs_arr = [1, 3, 4, 6, 8] #np.array(range(1,4)) #, 5, 6, 7] #LB and PP obs on 2018-07-(08, 09)
# obs_arr = [8, 9, 10, 11, 12, 13, 14] #LB and PP obs on 2018-07-(23, 24)
# obs_arr = [12, 13, 14] #LB and PP obs on 2018-07-(23, 24)

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[0] = 1 # uncomment and modify as required if adequate root of Gauss polynomial has to be selected

# a_mean, e_mean, taup_mean, I_mean, W_mean, w_mean = gm.gauss_method_mpc(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=5)

gm.gauss_method_sat(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=10)