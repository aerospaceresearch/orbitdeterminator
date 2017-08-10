'''
Server Version of Main.py
Runs the whole process in one file
Input a .csv positional data file (time, x, y, z) and this script generated the final set of keplerian elements
along with a plot and a filtered.csv data file
'''

import os
import numpy as np
import matplotlib as mpl
from subprocess import (PIPE, run)
import matplotlib.pylab as plt

from util import (read_data, kep_state, rkf78, golay_window)
from filters import (sav_golay, triple_moving_average)
from kep_determination import (lamberts_kalman, interpolation)



SOURCE_ABSOLUTE = os.getcwd() + "/src"  # Absolute path of source directory
os.system("cd %s; git init" % (SOURCE_ABSOLUTE))

def untracked_files():
    """Parses output of `git-status` and returns untracked files.

    Returns:
        res (string): List of untracked files.
    """
    res = run(
        "cd %s ; git status" % (SOURCE_ABSOLUTE),
        stdout=PIPE, stderr=PIPE,
        universal_newlines=True,
        shell=True
        )
    result = [line.strip() for line in res.stdout.split("\n")]

    files = [file
             for file in result if (file.endswith(".csv")
             and not (file.startswith("new file") or
             file.startswith("deleted") or file.startswith("modified")))]

    return files

def stage(processed):
    '''Stage the processed files into git file system

    Agrs:
        processed (list): List of processed files.
    '''
    for file in processed:
        print("staging")
        run(
            "cd %s;git add %s" % (SOURCE_ABSOLUTE, file),
            stdout=PIPE, stderr=PIPE,
            universal_newlines=True,
            shell=True
        )
        print("File %s has been staged." % (file))

def process(data_file):
    # First read the csv file called "orbit" with the positional data
    data = data_file
    # data = np.genfromtxt("orbit.csv")
    data = data[1:, :]
    data[:, 1:4] = data[:, 1:4] / 1000

    # Apply the Triple moving average filter with window = 3
    data_after_filter = triple_moving_average.generate_filtered_data(data, 3)


    ## Use the golay_window.py script to find the window for the savintzky golay filter based on the error you input
    error_apriori = 20  # input the a-priori error estimation for the data set
    c = golay_window.c(error_apriori)
    c = int(c)
    if (c % 2) == 0:
        c = c + 1
    window = len(data) / c
    window = int(window)


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


    # Save the filtered data into a new csv called "filtered"
    np.savetxt("filtered.csv", data_after_filter, delimiter=",")
    # Apply Lambert's solution for the filtered data set
    kep_lamb = lamberts_kalman.create_kep(data_after_filter)
    # Apply the interpolation method
    kep_inter = interpolation.main(data_after_filter)

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

    ## Finally we plot the graph
    # mpl.rcParams['legend.fontsize'] = 10
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(data[:, 1], data[:, 2], data[:, 3], ".", label='Initial data ')
    # ax.plot(data_after_filter[:, 1], data_after_filter[:, 2], data_after_filter[:, 3], "k", linestyle='-',
    # 		label='Filtered data')
    # ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label='Orbit after Interpolation method')
    # ax.legend()
    # ax.can_zoom()
    # ax.set_xlabel('x (km)')
    # ax.set_ylabel('y (km)')
    # ax.set_zlabel('z (km)')
    # plt.show()

def main():

    while True:
        raw_files = untracked_files()
        if not raw_files:
            pass
        else:
            for file in raw_files:
                print("processing")
                a = read_data.load_data(SOURCE_ABSOLUTE + "/" + file)
                process(a)
                print("File : %s has been processed \n \n" % a)
            stage(raw_files)

if __name__ == "__main__":
    main()