'''
Here we take the average of 3 terms x0, A, B where,
x0 = The point to be estimated
A = weighted average of n terms previous to x0
B = weighted avreage of n terms ahead of x0
n = window size
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from mpl_toolkits.mplot3d import Axes3D
from util import read_data as rd


def weighted_average(params):
    '''
    Calculates the weighted average of terms in the input

    Args:
        params (list): a list of numbers

    Returns:
        list: weighted average of the terms in the list
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
        signal_array (numpy array): the array of values on which the filter is to be applied
        window_size (int): the no. of points before and after x0 which should be considered for calculating A and B

    Returns:
       numpy array: a filtered array of size same as that of signal_array
    '''
    filtered_signal = []
    arr_len = len(signal_array)
    for index, point in enumerate(signal_array):
        if (index < window_size or index > arr_len - window_size ):
            filtered_signal.append(point)
        else:
            A, B = [], []
            for i in range(0, window_size):
                A.append(signal_array[index + i])
                B.append(signal_array[index - i])

            wa_A = weighted_average(A)
            wa_B = weighted_average(B)
            filtered_signal.append((point + wa_B + wa_A ) / 3)

    return filtered_signal

def generate_filtered_data(in_data, window):
    '''
    Apply the filter and generate the filtered data

    Args:
        in_data (string): numpy array containing the positional data
        window (int): window size applied into the filter

    Returns:
        numpy array: the final filtered array
    '''
    averaged_x = triple_moving_average(list(in_data[:,1]), window)
    averaged_y = triple_moving_average(list(in_data[:,2]), window)
    averaged_z = triple_moving_average(list(in_data[:,3]), window)

    output = np.hstack(((in_data[:,0])[:, np.newaxis], (np.array(averaged_x))[:, np.newaxis],
        (np.array(averaged_y))[:, np.newaxis], (np.array(averaged_z))[:, np.newaxis] ))

    return output

if __name__ == "__main__":

    signal = rd.load_data(os.getcwd() + '/' + sys.argv[1])

    import time
    time_start = time.clock()
    output = generate_filtered_data(signal, 2)
    time_stop = time.clock()

    print("File {} processed in {} seconds.".format(
            sys.argv[1],
            time_stop - time_start
        )
    )
    np.savetxt("filtered.csv", output, delimiter=",")

    print("Filtered output saved as filtered.csv")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(output[:,1], output[:,2], output[:,3], 'b', label='filtered')
    ax.plot(list(signal[:,1]), list(signal[:,2]), list(signal[:,3]), 'r', label='noisy')
    ax.legend(['Filtered Orbit', 'Noisy Orbit'])
    plt.show()
