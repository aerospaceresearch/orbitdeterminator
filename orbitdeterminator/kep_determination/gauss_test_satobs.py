import numpy as np
import gauss_method as gm

# path of file of optical IOD-formatted observations
# body_fname_str = '../example_data/SATOBS-BY-U-073118.txt'
body_fname_str = '../example_data/SATOBS-ML-19200716.txt'
# body_fname_str = '../example_data/iod_data_af2.txt'

#body name
# body_name_str = '18152 87 055A'
# body_name_str = '02142 66 031A'
body_name_str = 'ISS (25544)'

#lines of observations file to be used for orbit determination
# obs_arr = [1, 2, 3] # BY observations of 18152 87 055A on 2018 Jul 31st
# obs_arr = [1, 2, 3] # BY observations of 02142 66 031A on 2018 Aug 1st
# obs_arr = [1, 2, 3, 4, 5]
obs_arr = [1, 4, 6]

#the total number of observations used
nobs = len(obs_arr)

#select adequate index of Gauss polynomial root
r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[0] = 1 # uncomment and modify as required if adequate root of Gauss polynomial has to be selected

a, e, taup, I, W, w, T = gm.gauss_method_sat(body_fname_str, body_name_str, obs_arr, r2_root_ind_vec, refiters=10)

x = np.array((a, e, taup, I, W, w, T))

print('x = ', x)