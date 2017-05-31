'''
Author : Nilesh Chaturvedi
Date Created : 31st May, 2017

Triple Moving Average : Here we take the average of 3 terms x0, A, B where, 
x0 = The point to be estimated
A = weighted average of n terms previous to x0
B = weighted avreage of n terms ahead of x0
n = window size
'''

def triple_moving_average(signal_array, window_size):
    '''
    Apply triple moving average to a signal

    Args:
        signal_array : The array of values on which the filter is to be applied
        window_size : The no. of points before and after x0 which should be 
        considered for calculating A and B

    Returns:
       A filtered array of size same as that of signal_array 
    '''
    pass


if __name__ = "__main__":

    signal_x = []

    moving_window = 3

    averaged_signal = triple_moving_average(signal_x, moving_w)
