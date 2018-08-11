import numpy as np
import gauss_method as gm

# path of file of optical IOD-formatted observations
# body_fname_str = '../example_data/SATOBS-BY-U-073118.txt'
body_fname_str = '../example_data/SATOBS-ML-19200716.txt'

#body name
# body_name_str = '18152 87 055A'
body_name_str = 'ISS (25544)'

#lines of observations file to be used for orbit determination
# obs_arr = [1, 2, 3] # BY observations of 18152 87 055A on 2018 Jul 31st
obs_arr = [1, 4, 6]

###modify r2_root_ind_vec as necessary if adequate root of Gauss polynomial has to be selected
###if r2_root_ind_vec is not specified, then the first positive root will always be selected by default
# r2_root_ind_vec = zeros((len(obs_arr)-2,), dtype=int)
###select adequate index of Gauss polynomial root
# r2_root_ind_vec[4] = 1

a, e, taup, I, W, w, T = gm.gauss_method_sat(body_fname_str, body_name_str, obs_arr, refiters=10)
