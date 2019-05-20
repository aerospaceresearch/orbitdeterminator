'''
Server Version of Main.py
Runs the whole process in one file
Input a .csv positional data file (time, x, y, z) and this script generates the final set of keplerian elements
along with a plot and a filtered csv data file. Both the generated results lie in a folder named dst. 
'''

import os
import time
import sys
import numpy as np
import matplotlib as mpl
from subprocess import (PIPE, run)
import matplotlib.pylab as plt

from util import (read_data, kep_state, rkf78, golay_window)
from filters import (sav_golay, triple_moving_average)
from kep_determination import (lamberts_kalman, interpolation, gibbsMethod, ellipse_fit)
from propagation import sgp4


SOURCE_ABSOLUTE = os.getcwd() + "/example_data/SourceCSV"  # Absolute path of source directory
print("Do you wish to reset(deinit/init) git repository? [y/n]")
user_input1 = input()
if(user_input1 == "y" or user_input1 == "Y"):
    os.system("cd %s; rm -rf .git && rm -rf .gitignore" % (SOURCE_ABSOLUTE))
os.system("cd %s; git init" % (SOURCE_ABSOLUTE))


def untracked_files():
    '''
    Finds untracked/unprocessed files in the source directory.

    Returns:
        res (string): List of untracked files.
    '''

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
    '''
    Stage the processed files into git file system

    Agrs:
        processed (list): List of processed files.
    '''
    for file in processed:
        print("staging")
        run(
            "cd %s;git add '%s'" % (SOURCE_ABSOLUTE, file),
            stdout=PIPE, stderr=PIPE,
            universal_newlines=True,
            shell=True
        )
        print("File %s has been staged." % (file))

        
def process(data_file, error_apriori, name):
    ''' Perform filtering and orbit determination methods.
    Applies filters and orbit determination techniques on the input data and saves the 
    output in dst folder.
    
    Args:
        data_file (numpy array): Raw orbit data
        error_apriori (float): apriori estimation of the measurements error in km
        name (str): name of the file being processed
    '''
    # Get positional data
    data = data_file

    # Units is km by default

    # Apply the Triple moving average filter with window = 3
    data_after_filter = triple_moving_average.generate_filtered_data(data, 3)

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

    # Save the filtered data into a new csv called "filtered"
    np.savetxt(os.getcwd() + "/example_data/DestinationCSV/" + "%s_filtered.csv" % (name), data_after_filter, delimiter=",")

    # Apply Lambert's solution for the filtered data set
    kep_lamb = lamberts_kalman.create_kep(data_after_filter)

    # Apply the interpolation method
    kep_inter = interpolation.main(data_after_filter)

    # Apply the Gibbs method
    kep_gibbs = gibbsMethod.gibbs_get_kep(data_after_filter[:,1:])

    # Apply the ellipse best fit method
    kep_ellip = ellipse_fit.determine_kep(data_after_filter[:,1:])[0]

    # Apply Kalman filters to find the best approximation of the keplerian elements for all solutions
    # set we a estimate of measurement vatiance R = 0.01 ** 2
    kep_final_lamb = lamberts_kalman.kalman(kep_lamb, 0.01 ** 2)
    kep_final_lamb = np.transpose(kep_final_lamb)
    kep_final_lamb = np.resize(kep_final_lamb, ((7, 1)))
    kep_final_lamb[6, 0] = sgp4.rev_per_day(kep_final_lamb[0, 0])

    kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)
    kep_final_inter = np.transpose(kep_final_inter)
    kep_final_inter = np.resize(kep_final_inter, ((7, 1)))
    kep_final_inter[6, 0] = sgp4.rev_per_day(kep_final_inter[0, 0])

    kep_final_ellip = np.transpose(kep_ellip)
    kep_final_ellip = np.resize(kep_final_ellip, ((7, 1)))
    kep_final_ellip[6, 0] = sgp4.rev_per_day(kep_final_ellip[0, 0])

    kep_final_gibbs = lamberts_kalman.kalman(kep_gibbs, 0.01 ** 2)
    kep_final_gibbs = np.transpose(kep_final_gibbs)
    kep_final_gibbs = np.resize(kep_final_gibbs, ((7, 1)))
    kep_final_gibbs[6, 0] = sgp4.rev_per_day(kep_final_gibbs[0, 0])

    kep_final = np.zeros((7, 4))
    kep_final[:, 0] = np.ravel(kep_final_lamb)
    kep_final[:, 1] = np.ravel(kep_final_inter)
    kep_final[:, 2] = np.ravel(kep_final_ellip)
    kep_final[:, 3] = np.ravel(kep_final_gibbs)

    # Print the final orbital elements for all solutions
    kep_elements = ["Semi major axis (a)(km)", "Eccentricity (e)", "Inclination (i)(deg)", "Argument of perigee (ω)(deg)", "Right acension of ascending node (Ω)(deg)", "True anomaly (v)(deg)", "Frequency (f)(rev/day)"]
    det_methods = ["Lamberts Kalman", "Spline Interpolation", "Ellipse Best Fit", "Gibbs 3 Vector"]
    method_name = ["lamb", "inter", "ellip", "gibb"]
    
    for i in range(0, 4):
        print("\n******************Output for %s Method******************\n" % det_methods[i])
        j = 0
        for j in range(0, 7):
            print("%s: %.16f\n" % (kep_elements[j], kep_final[j, i]))

    print("\nSave plots? [y/n]")
    user_input = input()

    if(user_input == "y" or user_input == "Y"):
        for j in range(0, 4):
            # Plot the initial data set, the filtered data set and the final orbit
            # First we transform the set of keplerian elements into a state vector
            state = kep_state.kep_state(np.resize(kep_final[:, j], (7, 1)))

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
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot(data[:, 1], data[:, 2], data[:, 3], ".", label='Initial data ')
            ax.plot(data_after_filter[:, 1], data_after_filter[:, 2], data_after_filter[:, 3], "k", linestyle='-',
                    label='Filtered data')
            ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label='Orbit after %s method' % det_methods[j])
            ax.legend()
            ax.can_zoom()
            ax.set_xlabel('x (km)')
            ax.set_ylabel('y (km)')
            ax.set_zlabel('z (km)')
            plt.savefig(os.getcwd() + "/example_data/DestinationSVG/" + '%s_%s.svg'%(name, method_name[j]), format="svg")
            print("saved %s_%s.svg"%(name, method_name[j]))

def main():

    number_untracked = 0
    while True:
        raw_files = untracked_files()
        if not raw_files:
            if (number_untracked == 0):
                print("\nNo unprocessed file found in ./example_data/SourceCSV folder")
            else:
                print("\nAll untracked files have been processed")
            print("Add new files in ./example_data/SourceCSV folder to process them")
            time_elapsed = 0
            timeout = 30
            while (time_elapsed <= timeout and not raw_files):
                sys.stdout.write("\r")
                sys.stdout.write("-> Timeout in - {:2d} s".format(timeout - time_elapsed))
                sys.stdout.flush()
                time.sleep(1)
                time_elapsed += 1
                raw_files = untracked_files()
            sys.stdout.write("\r                        \n")
            pass
        if raw_files:
            number_untracked += len(raw_files)
            for file in raw_files:
                print("processing")
                a = read_data.load_data(SOURCE_ABSOLUTE + "/" + file)

                process(a, 10.0, str(file)[:-4])
                print("File : %s has been processed \n \n" % file)
            stage(raw_files)
            continue
        print("No new unprocessed file was added, program is now exiting due to timeout!")
        print("Total {} untracked files were processed".format(number_untracked))
        break

if __name__ == "__main__":
    main()
