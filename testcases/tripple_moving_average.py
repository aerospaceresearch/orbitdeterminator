'''
Author : Nilesh Chaturvedi
Date Created : 31st May, 2017

Triple Moving Average : Here we take the average of 3 terms x0, A, B where,
x0 = The point to be estimated
A = weighted average of n terms previous to x0
B = weighted avreage of n terms ahead of x0
n = window size
'''
import os
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import read_data as rd


def weighted_average(params):
    '''Calculates the weighted average of terms in the input.

    Args:
        params: a list of numbers.

    Returns:
        weighted average of the terms in the list
    '''
    weighted_sum = 0
    weight = len(params)
    weight_sum = weight * (weight+1) / 2
    for num in params:
        weighted_sum += weight*num
        weight -= 1

    return weighted_sum / weight_sum

def extrapolation_padding(path, window):
    '''Pad the input with approximate values. To be called with size of data being
    equal to the window size.

    Args:
        path: path matrix.
        window: window size to be used in triple moving average.

    Returns:
        extrapolated: 3D path matrix with extrapolated padding on either sides of the path.
    '''
    size = len(path)-1

    path_x = path[:, 1]
    path_y = path[:, 2]
    path_z = path[:, 3]

    pre_factor_x = path_x[1]/path_x[2]
    pre_factor_y = path_y[1]/path_y[2]
    pre_factor_z = path_z[1]/path_z[2]
    post_factor_x = path_x[size]/path_x[size-1]
    post_factor_y = path_y[size]/path_y[size-1]
    post_factor_z = path_z[size]/path_z[size-1]

    pre_x, pre_y, pre_z, post_x, post_y, post_z = [], [], [], [], [], []

    for index in range(window):
        pre_x.append(path_x[index+1]*(pre_factor_x ** (index+1)))
        post_x.append(path_x[size]*(post_factor_x ** (index+1)))
        pre_y.append(path_y[index+1]*(pre_factor_y ** (index+1)))
        post_y.append(path_y[size]*(post_factor_y ** (index+1)))    
        pre_z.append(path_z[1]*(pre_factor_z ** (index+1)))
        post_z.append(path_z[size]*(post_factor_z ** (index+1)))

    pre_x = np.flipud(pre_x)
    pre_y = np.flipud(pre_y)
    pre_x = np.flipud(pre_x)

    new_x = np.concatenate((pre_x, path_x, post_x), axis =0)
    new_y = np.concatenate((pre_y, path_y, post_y), axis =0)
    new_z = np.concatenate((pre_z, path_z, post_z), axis =0)

    extrapolated = np.hstack(((new_x)[:, np.newaxis], (new_y)[:, np.newaxis], (new_z)[:, np.newaxis]))

    return extrapolated

def triple_moving_average(signal_array, window_size):
    '''Apply triple moving average to a signal.

    Args:
        signal_array: the array of values on which the filter is to be applied.
        window_size: the no. of points before and after x0 which should be
        considered for calculating A and B.

    Returns:
       A filtered array of size same as that of signal_array.
    '''
    filtered_signal = []
    arr_len = len(signal_array)
    for point in signal_array[ window_size - 1 : arr_len - window_size -1 ]:
        # if (signal_array.index(point) < window_size or signal_array.index(point) > arr_len - window_size ):
        #     filtered_signal.append(point)
        # else:
        A, B = [], []
        pos = signal_array.index(point)
        for i in range(1, window_size):
            A.append(signal_array[pos + i])
            B.append(signal_array[pos - i])

        wa_A = weighted_average(A)
        wa_B = weighted_average(B)
        filtered_signal.append((point + wa_B + wa_A ) / 3)

    return filtered_signal

def generate_filtered_data(file, window):
    '''Apply filtering to individual co-ordinates

    Args:
        file: 4D orbit data [time, x, y, z].
        window: window size to be used in triple moving average.

    Returns:
        output: 4D filtered orbit data [time, x, y, z].
    '''
    data = extrapolation_padding(file, window)
    averaged_x = (triple_moving_average(list(data[:,0]), window))
    averaged_y = triple_moving_average(list(data[:,1]), window)
    averaged_z = triple_moving_average(list(data[:,2]), window)

    print(len(averaged_y))
    print(len(file[:, 0]))
    output = np.hstack(((file[:,0])[:, np.newaxis], (np.array(averaged_x))[:, np.newaxis],
        (np.array(averaged_y))[:, np.newaxis], (np.array(averaged_z))[:, np.newaxis] ))

    return output

if __name__ == "__main__":

    signal = rd.load_data(os.getcwd() + '/' + sys.argv[1])

    output = generate_filtered_data(signal, 5)
    np.savetxt("filtered.csv", output, delimiter=",")

    print("Filtered output saved as filtered.csv")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(output[:,1], output[:,2], output[:,3], 'b', label='filtered')
    ax.scatter(list(signal[:,1]), list(signal[:,2]), list(signal[:,3]), 'r', label='noisy')
    ax.legend(['Filtered Orbit', 'Noisy Orbit'])
    plt.show()
