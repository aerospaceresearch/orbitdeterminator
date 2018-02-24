"""
Here we take the average of 3 terms x0, A, B where,
x0 = The point to be estimated
A = weighted average of n terms previous to x0
B = weighted avreage of n terms ahead of x0
n = window size
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# from util import read_data as rd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def distance_weighted_average(values_array: np.ndarray, direction='f'):
    """Calculates the weighted average in terms of distance from origin.

    Most weight is assigned to first element of array, every subsequent element
    reduces in weight.

    Args:
        values_array (numpy array): a list of numbers
        direction: f for forward distance weight (smaller index elements have heavier weights), or r for reverse

    Returns:
        weighted average of the values in the list
    """

    assert direction == 'f' or direction == 'r'

    if direction == 'f':
        weights = range(len(values_array), 0, -1)
    else:
        weights = range(1, len(values_array) + 1)

    weights = np.asarray(weights)
    weighted_sum = sum(values_array * weights)

    return weighted_sum / sum(weights)


def smooth_moving_average(signal_array, window_size):
    """Apply moving average to smooth out a signal.

    Args:
        signal_array (numpy array): the array of values on which the filter is to be applied
        window_size (int): number of points in the moving average. Points included are x0 (middle point) and
                           floor(window_size / 2) points before and after x0. Must be impair.

    Returns:
       numpy array: a filtered array of size same as that of signal_array
    """

    assert window_size % 2 == 1

    array_len = len(signal_array)
    window_wing = int(np.floor(window_size / 2))
    filtered_signal = np.zeros(array_len)

    for curr_index in range(array_len):
        if curr_index < window_wing + 1 or curr_index > array_len - window_wing - 2:
            filtered_signal[curr_index] = signal_array[curr_index]
        else:
            head = signal_array[curr_index - window_wing: curr_index + 1]
            tail = signal_array[curr_index: curr_index + window_wing + 1]
            wa_head = distance_weighted_average(head, 'r')
            wa_tail = distance_weighted_average(tail)

            filtered_signal[curr_index] = (wa_head + signal_array[curr_index] + wa_tail) / 3

    return filtered_signal


def generate_filtered_data(filename, window):
    '''
    Apply the filter and generate the filtered data

    Args:
        filename (string): the name of the .csv file containing the positional data
        window (int): window size applied into the filter

    Returns:
        numpy array: the final filtered array
    '''
    averaged_x = (smooth_moving_average(list(filename[:, 1]), window))
    averaged_y = smooth_moving_average(list(filename[:, 2]), window)
    averaged_z = smooth_moving_average(list(filename[:, 3]), window)

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