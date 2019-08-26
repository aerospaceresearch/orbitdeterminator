#!/usr/bin/env python3
'''
Runs the whole process in one file for a .csv positional data file (time, x, y, z)
and generates the final set of keplerian elements along with a plot and a filtered.csv data file
'''

import sys
import os
sys.path.append(os.getcwd())
import warnings
from util import read_data, kep_state, rkf78, golay_window, get_format, convert_format, handle_multiple_files
from filters import sav_golay, triple_moving_average
from kep_determination import lamberts_kalman, interpolation, ellipse_fit, gibbsMethod
import automated
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from propagation import sgp4
import inquirer

def process(file_list, error_apriori, units, output):
    '''
    Given a .csv data file in the format of (time, x, y, z) applies both filters, generates a filtered.csv data
    file, prints out the final keplerian elements computed from both Lamberts and Interpolation and finally plots
    the initial, filtered data set and the final orbit.

    Args:
        file_list (string): List of strings containing file names
        error_apriori (float): apriori estimation of the measurements error in km

    Returns:
        Runs the whole process of the program
    '''
    source_path = "obs_data/Source"
    dest_path = "obs_data/DestinationCSV/"
    data = handle_multiple_files.handle_multiple_files(file_list)

    print("***********Choose from following options in desired order of application***********")
    print("(SPACE to toggle, UP/DOWN to navigate, RIGHT/LEFT to select/deselect and ENTER to submit)")
    print("*Default options are selected below, if nothing is selected these options will be executed")
    questions = [
      inquirer.Checkbox('filter',
                        message="Select filter(s)",
                        choices=['Triple Moving Average Filter', 'Savitzky Golay Filter'],
                        default=['Triple Moving Average Filter', 'Savitzky Golay Filter'],
                        ),
      inquirer.Checkbox('method',
                        message="Select Method(s)",
                        choices=['Lamberts Kalman', 'Cubic Spline Interpolation', 'Ellipse Best Fit', 'Gibbs 3 Vector'],
                        default=['Cubic Spline Interpolation'],
                        ),
    ]
    choices = inquirer.prompt(questions)

    if(units == 'm'):
        # Transform m to km
        data[:, 1:4] = data[:, 1:4] / 1000
    data_after_filter = data

    if not choices['filter']:
        print("Applying Triple Moving Average followed by Savitzky Golay...")
        choices['filter'].append('Triple Moving Average Filter')
        choices['filter'].append('Savitzky Golay Filter')
        # Apply the Triple moving average filter with window = 3
        data_after_filter = triple_moving_average.generate_filtered_data(data_after_filter, 3)

        # Use the golay_window.py script to find the window for the Savitzky Golay filter based on the error you input
        window = golay_window.window(error_apriori, data_after_filter)
        
        # Apply the Savitzky Golay filter with window = window (51 for orbit.csv) and polynomial order = 3
        data_after_filter = sav_golay.golay(data_after_filter, window, 3)
    else:
        print(choices['filter'])
        for index, choice in enumerate(choices['filter']):
            if(choice == 'Savitzky Golay Filter'):
                print("Applying Savitzky Golay Filter...")
                # Use the golay_window.py script to find the window for the Savitzky Golay filter based on the error you input
                window = golay_window.window(error_apriori, data_after_filter)

                # Apply the Savitzky Golay filter with window = window (51 for orbit.csv) and polynomial order = 3
                data_after_filter = sav_golay.golay(data_after_filter, window, 3)
            else:
                print("Applying Triple Moving Average Filter...")
                # Apply the Triple moving average filter with window = 3
                data_after_filter = triple_moving_average.generate_filtered_data(data_after_filter, 3)

    # Compute the residuals between filtered data and initial data and then the sum and mean values of each axis
    res = data_after_filter[:, 1:4] - data[:, 1:4]
    sums = np.sum(res, axis=0)
    print("\nDisplaying the sum of the residuals for each axis")
    print(sums, "\n")

    means = np.mean(res, axis=0)
    print("Displaying the mean of the residuals for each axis")
    print(means, "\n")

    # Save the filtered data into a new csv called "filtered"
    np.savetxt(dest_path + output + ".csv", data_after_filter, delimiter=",")

    kep_elements = {}

    if not choices['method']:
        choices['method'].append('Cubic Spline Interpolation')
        # Apply the interpolation method
        kep_inter = interpolation.main(data_after_filter)
        # Apply Kalman filters to find the best approximation of the keplerian elements for all solutions
        # We set an estimate of measurement variance R = 0.01 ** 2
        kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)
        kep_final_inter = np.transpose(kep_final_inter)
        kep_final_inter = np.resize(kep_final_inter, ((7, 1)))
        kep_final_inter[6, 0] = sgp4.rev_per_day(kep_final_inter[0, 0])
        kep_elements['Cubic Spline Interpolation'] = kep_final_inter
    else:
        for index, choice in enumerate(choices['method']):
            if(choice == 'Lamberts Kalman'):
                # Apply Lambert Kalman method for the filtered data set
                kep_lamb = lamberts_kalman.create_kep(data_after_filter)
                # Apply Kalman filters to find the best approximation of the keplerian elements for all solutions
                # We set an estimate of measurement variance R = 0.01 ** 2
                kep_final_lamb = lamberts_kalman.kalman(kep_lamb, 0.01 ** 2)
                kep_final_lamb = np.transpose(kep_final_lamb)
                kep_final_lamb = np.resize(kep_final_lamb, ((7, 1)))
                kep_final_lamb[6, 0] = sgp4.rev_per_day(kep_final_lamb[0, 0])
                kep_elements['Lamberts Kalman'] = kep_final_lamb
            elif(choice == 'Cubic Spline Interpolation'):
                # Apply the interpolation method
                kep_inter = interpolation.main(data_after_filter)
                # Apply Kalman filters to find the best approximation of the keplerian elements for all solutions
                # We set an estimate of measurement variance R = 0.01 ** 2
                kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)
                kep_final_inter = np.transpose(kep_final_inter)
                kep_final_inter = np.resize(kep_final_inter, ((7, 1)))
                kep_final_inter[6, 0] = sgp4.rev_per_day(kep_final_inter[0, 0])
                kep_elements['Cubic Spline Interpolation'] = kep_final_inter
            elif(choice == 'Ellipse Best Fit'):
                # Apply the ellipse best fit method
                kep_ellip = ellipse_fit.determine_kep(data_after_filter[:, 1:])[0]
                kep_final_ellip = np.transpose(kep_ellip)
                kep_final_ellip = np.resize(kep_final_ellip, ((7, 1)))
                kep_final_ellip[6, 0] = sgp4.rev_per_day(kep_final_ellip[0, 0])
                kep_elements['Ellipse Best Fit'] = kep_final_ellip
            else:
                # Apply the Gibbs method
                kep_gibbs = gibbsMethod.gibbs_get_kep(data_after_filter[:,1:])
                # Apply Kalman filters to find the best approximation of the keplerian elements for all solutions
                # We set an estimate of measurement variance R = 0.01 ** 2
                kep_final_gibbs = lamberts_kalman.kalman(kep_gibbs, 0.01 ** 2)
                kep_final_gibbs = np.transpose(kep_final_gibbs)
                kep_final_gibbs = np.resize(kep_final_gibbs, ((7, 1)))
                kep_final_gibbs[6, 0] = sgp4.rev_per_day(kep_final_gibbs[0, 0])
                kep_elements['Gibbs 3 Vector'] = kep_final_gibbs

    kep_final = np.zeros((7, len(kep_elements)))
    order = []
    for index, key in enumerate(kep_elements):
        kep_final[:, index] = np.ravel(kep_elements[key])
        order.append(str(key))

    # Print the final orbital elements for all solutions
    kep_elements = ["Semi major axis (a)(km)", "Eccentricity (e)", "Inclination (i)(deg)", "Argument of perigee (ω)(deg)", \
                    "Right acension of ascending node (Ω)(deg)", "True anomaly (v)(deg)", "Frequency (f)(rev/day)"]
    for i in range(0, len(order)):
        print("\n******************Output for %s Method******************\n" % order[i])
        for j in range(0, 7):
            print("%s: %.8f" % (kep_elements[j], kep_final[j, i]))
    print("\n")

    order = choices['method']
    print("***********Select if you want to plot the orbit***********")
    print("(Press ENTER to select)")
    questions = [
      inquirer.List('plots',
                        message="Plot orbit?",
                        choices=['Yes', 'No'],
                        ),
    ]
    choices = inquirer.prompt(questions)
    if (choices['plots'] == 'No'):
        pass
    else:
        print("***********Choose methods for which orbit plots are desired***********")
        print("(SPACE to toggle, UP/DOWN to navigate, RIGHT/LEFT to select/deselect and ENTER to submit)")
        questions = [
          inquirer.Checkbox('plots',
                            message="Select methods(s)",
                            choices=order,
                            ),
        ]
        choices = inquirer.prompt(questions)

        if not choices['plots']:
            choices['plots'].append(order[0])
        else:
            pass

        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(data[:, 1], data[:, 2], data[:, 3], ".", label='Initial data ')
        ax.plot(data_after_filter[:, 1], data_after_filter[:, 2], data_after_filter[:, 3], "k", linestyle='-',
                label='Filtered data')
        for j in range(len(choices['plots'])):
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
            ax.plot(positions[0, :], positions[1, :], positions[2, :], label='Orbit using %s method' % order[j])
        ax.legend()
        ax.can_zoom()
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_zlabel('z (km)')
        plt.show()

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, nargs='*', help="Path to .csv data file", default=['orbit.csv'])
    parser.add_argument('-e', '--error', type=float, help="Estimation of the measurement error", default=10.0)
    parser.add_argument('-u', '--units', type=str, help="m for metres, k for kilometres", default='k')
    parser.add_argument('-o', '--output', type=str, help="Filename to save filtered data in DestinationCSV folder", default='filtered')
    parser.add_argument('-a', '--automate', help="Automate the orbit determination process", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    print("\n************Welcome To OrbitDeterminator************\n")
    print("Workflow for OrbitDeterminator is as follows:")
    workflow = "                   -----------    ----------------------\n"\
               "Positional data--->| Filters |--->| Keplerian elements |--->Determined Orbit\n"\
               "                   |         |    | Determination      |\n"\
               "                   -----------    ----------------------\n\n"\
               "Available filters:               | Available methods for orbit determination:\n"\
               "  1. Savitzky Golay Filter       |   1. Lamberts Kalman\n"\
               "  2. Triple Moving Average Filter|   2. Cubic spline interpolation\n"\
               "                                 |   3. Ellipse Bset Fit\n"\
               "                                 |   4. Gibbs 3 Vector\n"
    print("\n" + workflow)
    args = read_args()
    if args.automate:
        automated.main()
    else:
        process(args.file_path, args.error, args.units, args.output)
