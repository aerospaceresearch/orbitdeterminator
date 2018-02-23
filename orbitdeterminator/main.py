"""
Runs the whole process in one file for a .csv positional data file (time, x, y, z)
and generates the final set of keplerian elements along with a plot and a filtered.csv data file
"""


import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from util import (read_data, kep_state, rkf78, golay_window)
from filters import (sav_golay, smooth_moving_average)
from kep_determination import (lamberts_kalman, interpolation)


def apply_filters(data, error_apriori, save=True):
    # Apply the smoothing moving average filter with window = 5
    data_after_filter = smooth_moving_average.generate_filtered_data(data, 5)

    # Use the golay_window.py script to find the window for the savintzky golay filter based on the error you input
    window = golay_window.window(error_apriori, data_after_filter)

    # Apply the Savintzky - Golay filter with window = 31 and polynomail parameter = 6
    data_after_filter = sav_golay.golay(data_after_filter, window, 3)

    # Compute the residuals between filtered data and initial data and then the sum and mean values of each axis
    res = data_after_filter[:, 1:4] - data[:, 1:4]
    sums = np.sum(res, axis=0)
    print("Displaying the sum of the residuals for each axis")
    print(sums)
    print(" ")

    means = np.mean(res, axis=0)
    print("Displaying the mean of the residuals for each axis")
    print(means)
    print(" ")

    if save:
        # Save the filtered data into a new csv called "filtered"
        np.savetxt("filtered.csv", data_after_filter, delimiter=",")

    return data_after_filter


def process(data_file, error_apriori):
    """
    Given a .csv data file in the format of (time, x, y, z), applies both filters, generates a filtered.csv data
    file, prints out the final keplerian elements computed from both Lamberts and Interpolation and finally plots
    the initial, filtered data set and the final orbit.

    Args:
        data_file (string): The name of the .csv file containing the positional data
        error_apriori (float): apriori estimation of the measurements error in km

    Returns:
        Runs the whole process of the program
    """
    # Read the csv file with positional data
    data = read_data.load_data(data_file)

    # Transform m to km
    data[:, 1:4] = data[:, 1:4] / 1000

    filtered_data = apply_filters(data, error_apriori)

    # Apply Lambert's solution for the filtered data set
    kep_lamb = lamberts_kalman.create_kep(filtered_data)

    # Apply the interpolation method
    kep_inter = interpolation.main(filtered_data)

    # Apply Kalman filters to find the best approximation of the keplerian elements for both solutions
    # set we a estimate of measurement vatiance R = 0.01 ** 2
    kep_final_lamb = lamberts_kalman.kalman(kep_lamb, 0.01 ** 2)
    kep_final_lamb = np.transpose(kep_final_lamb)

    kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)
    kep_final_inter = np.transpose(kep_final_inter)

    kep_final_lamb[5, 0] = kep_final_inter[5, 0]

    kep_final = np.zeros((6, 2))
    kep_final[:, 0] = np.ravel(kep_final_lamb)
    kep_final[:, 1] = np.ravel(kep_final_inter)

    # Print the final orbital elements for both solutions
    print("Displaying the final keplerian elements, first row : Lamberts, second row : Interpolation")
    print(kep_final)

    # Plot the initial data set, the filtered data set and the final orbit

    # First we transform the set of keplerian elements into a state vector
    state = kep_state.kep_state(kep_final_inter)

    # Then we produce more state vectors at varius times using a Runge Kutta algorithm
    keep_state = np.zeros((6, 150))
    ti = 0.0
    tf = 1.0
    t_hold = np.zeros((150, 1))
    x = state
    h = 0.1
    tetol = 1e-04
    for i in range(0, 150):
        keep_state[:, i] = np.ravel(rkf78.rkf78(6, ti, tf, h, tetol, x))
        t_hold[i, 0] = tf
        tf = tf + 1

    positions = keep_state[0:3, :]

    # Finally we plot the graph
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(data[:, 1], data[:, 2], data[:, 3], ".", label='Initial data ')
    ax.plot(filtered_data[:, 1], filtered_data[:, 2], filtered_data[:, 3], "k", linestyle='-',
            label='Filtered data')
    ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label='Orbit after Interpolation method')
    ax.legend()
    ax.can_zoom()
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')
    plt.show()


def read_args():
    """Fetches commandline arguments.

    Retrieves the optional file path and error term commandline arguments. Provides 'orbit.csv' and 10.0
    as defaults.

    Returns:
        Parsed arguments file path (str) and error term (float).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str,
                        help="path to .csv data file",
                        default='orbit.csv')
    parser.add_argument('-e', '--error', type=float,
                        help="estimation of the measurement error",
                        default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    process(args.file_path, args.error)
