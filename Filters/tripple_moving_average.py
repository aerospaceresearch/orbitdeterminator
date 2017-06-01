'''
Author : Nilesh Chaturvedi
Date Created : 31st May, 2017

Triple Moving Average : Here we take the average of 3 terms x0, A, B where, 
x0 = The point to be estimated
A = weighted average of n terms previous to x0
B = weighted avreage of n terms ahead of x0
n = window size
'''
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
    for point in signal_array:
        A = []
        B = []
        while(window_size !=0 ):




    pass


if __name__ == "__main__":

    signal_x = []

    moving_window = 3

    averaged_signal = triple_moving_average(signal_x, moving_window)