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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import read_data as rd


def weighted_average(params):
    '''
    Calculates the weighted average of terms in the input

    Args:
        params: a list of numbers

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


def triple_moving_average(signal_array, window_size):
    '''
    Apply triple moving average to a signal

    Args:
        signal_array: the array of values on which the filter is to be applied
        window_size: the no. of points before and after x0 which should be
        considered for calculating A and B

    Returns:
       A filtered array of size same as that of signal_array
    '''
    filtered_signal = []
    arr_len = len(signal_array)
    for point in signal_array:
        if (signal_array.index(point) < window_size or signal_array.index(point) > arr_len - window_size ):
            filtered_signal.append(point)
        else:
            A, B = [], []
            pos = signal_array.index(point)
            for i in range(0, window_size):
                A.append(signal_array[pos + i])
                B.append(signal_array[pos - i])

            wa_A = weighted_average(A)
            wa_B = weighted_average(B)
            filtered_signal.append((point + wa_B + wa_A ) / 3)

    return filtered_signal

def generate_filtered_data(filename, window):

    averaged_x = (triple_moving_average(list(filename[:,1]), window))
    averaged_y = triple_moving_average(list(filename[:,2]), window)
    averaged_z = triple_moving_average(list(filename[:,3]), window)

    output = np.hstack(((filename[:,0])[:, np.newaxis], (np.array(averaged_x))[:, np.newaxis],
        (np.array(averaged_y))[:, np.newaxis], (np.array(averaged_z))[:, np.newaxis] ))

    return output

if __name__ == "__main__":

    signal = rd.load_data(os.getcwd() + '/' + sys.argv[1])
    
    output = generate_filtered_data(signal, 2)
    np.savetxt("filtered.csv", output, delimiter=",")

    print("Filtered output saved as filtered.csv")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(output[:,1], output[:,2], output[:,3], 'b', label='filtered')
    ax.plot(list(signal[:,1]), list(signal[:,2]), list(signal[:,3]), 'r', label='noisy')
    ax.legend(['Filtered Orbit', 'Noisy Orbit'])
    plt.show()