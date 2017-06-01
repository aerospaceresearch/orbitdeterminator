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
    arr_len = len(signal_array)
    
    for point in signal_array[]:
        if (signal_array.index(point) < window_size or signal_array.index(point) > arr_len - window_size ):
            filtered_signal.append(point)

        else:
            A, B = [], []
            pos = signal_array.index(point)
            
            for i in range(1, window_size):
                A.append[signal_array[pos + i]]
                b.append[signal_array[pos - i]]

            wa_A = weighted_average(A)
            wa_B = weighted_average(B)
            filtered_signal.append((point + wa_B + wa_A ) / 3)

    return filtered_signal





if __name__ == "__main__":

    signal_x = []

    moving_window = 3

    averaged_signal = triple_moving_average(signal_x, moving_window)