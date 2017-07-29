"""
Author : Nilesh Chaturvedi
Date Created : 12th June, 2017

Analysis of varying parameres with filters in sequence
"""
import os
import numpy
from scipy.signal import savgol_filter

import read_data as rd
import tripple_moving_average as tma
import interpolation
import pickle


# def process(file):
#     signal = rd.load_data(os.getcwd() + "/track/" + file)
#     for tma_window in range(2, 9):
#         print(tma_window)
#         tma_filtered = tma.generate_filtered_data(signal, tma_window)
#         for golay_degree in range(1, 9):
#             print (golay_degree)
#             for golay_window in range(9, 99, 2):
#                 print(golay_window)
#                 print(file)
#                 output = open("result.txt", 'a')
#                 golay_filtered = golay(tma_filtered, golay_window, golay_degree)
#                 param_state = tuple([file, tma_window, golay_window, golay_degree])
#                 a = file[16:30]
#                 kep = interpolation.main(golay_filtered)  # Keplerian determination happens here. Change this accordingly
#                 error = abs(kep_dict[a] - kep).mean(axis=0)
#                 output.write(str(param_state) + " : " + str(error))
#                 output.close()

#                 final_dict[param_state] = error

def golay(data, window, degree):

    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    x_new = savgol_filter(x, window, degree)
    y_new = savgol_filter(y, window, degree)
    z_new = savgol_filter(z, window, degree)

    new_positions = numpy.zeros((len(data), 4))
    new_positions[:, 1] = x_new
    new_positions[:, 2] = y_new
    new_positions[:, 3] = z_new
    new_positions[:, 0] = data[:, 0]

    return new_positions

######################################################################################
noisy_path = os.getcwd() + "/track1/"
noisy_files = os.listdir(noisy_path)
true_keps = open("true_kep_dict.p", 'rb')
kep_dict = pickle.load(true_keps)
print(kep_dict)
final_dict = {}
######################################################################################
# noise_0, noise_10000, noise_20000, noise_40000, noise_80000 = [], [], [], [], []
# [noise_0.append(i) for i in noisy_files if i.endswith("0.csv")]
# [noise_10000.append(i) for i in noisy_files if i.endswith("10000.csv")]
# [noise_20000.append(i) for i in noisy_files if i.endswith("20000.csv")]
# [noise_40000.append(i) for i in noisy_files if i.endswith("40000.csv")]
# [noise_80000.append(i) for i in noisy_files if i.endswith("80000.csv")]
######################################################################################

for file in noisy_files:
    signal = rd.load_data(os.getcwd() + "/track1/" + file)
    for tma_window in range(2, 9):
        print(tma_window)
        tma_filtered = tma.generate_filtered_data(signal, tma_window)
        for golay_degree in range(1, 9):
            print (golay_degree)
            for golay_window in range(9, 99, 2):
                print(golay_window)
                print(file)
                output = open("result.txt", 'a')
                golay_filtered = golay(tma_filtered, golay_window, golay_degree)
                param_state = tuple([file, tma_window, golay_window, golay_degree])
                a = file[16:30]
                kep = interpolation.main(golay_filtered)  # Keplerian determination happens here. Change this accordingly
                error = abs(kep_dict[a] - kep).mean(axis=0)
                output.write(str(param_state) + " : " + str(error))
                output.close()

                final_dict[param_state] = error

output.close()
pickle.dump(final_dict, open("final_dict.p", 'wb'))

#errors = numpy.array(errors)
#errors = errors[numpy.argsort(errors[:,3])]

# print("The best filter combination has the following configuration"
#     "\n\nTriple Moving Average Window {} \nGolay Window {} \nGolay Degree {}"
#     "\nError {}".format(errors[0][0], errors[0][1], errors[0][2], errors[0][3]))
