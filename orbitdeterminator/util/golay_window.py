def c(error):
    '''Calculates the constant c which is needed to determine the savintzky - golay filter window
    window = len(data) / c ,where c is a constant strongly related to the error contained in the data set

    Args:
        error(float): the a-priori error estimation for each measurment

    Returns:
        float: constant which describes the window that needs to be inputed to the savintzky - golay filter

    '''
    if error <= 40.0:
        c = 10.26 + (10676069.73 / (1 + ((error/1.242)**5.367)))
    else:
        c = (- 0.046725 * error) + 13.102

    return c


if __name__ == "__main__":
    x_error = 10   # x_error = 10 means, 10km a-priori error estimation, for points with time difference of 1 second
    print(c(x_error))
