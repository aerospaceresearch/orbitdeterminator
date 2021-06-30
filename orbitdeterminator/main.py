'''
Runs the whole process in one file for a .csv positional data file (time, x, y, z)
and generates the final set of keplerian elements along with a plot and a filtered.csv data file
'''


from util import (read_data, kep_state, rkf78, golay_window)
from filters import (sav_golay, triple_moving_average, wiener)
from kep_determination import (lamberts_kalman, interpolation, ellipse_fit, gibbs_method, gauss_method)
from optimization import (with_mcmc)
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from propagation import sgp4
import inquirer
from vpython import *
import animate_orbit
import kep_determination.orbital_elements as oe
import random


def get_timestamp_index_by_orbitperiod(semimajor_axis, timestamps):

    T_orbitperiod = oe.T_orbitperiod(semimajor_axis=semimajor_axis)
    runtime = np.subtract(timestamps, np.min(timestamps))
    index = np.argmax(runtime >= T_orbitperiod // 2) - 1  # only half orbit is good for Gibbs method

    if index < 2:
        # in case there are not enough points to have the result at index point at 2
        # or the argmax search does not find anything and sets index = 0.
        index = len(timestamps) - 1

    return index


def process(data_file, error_apriori, units):
    '''
    Given a .csv data file in the format of (time, x, y, z) applies both filters, generates a filtered.csv data
    file, prints out the final keplerian elements computed from both Lamberts and Interpolation and finally plots
    the initial, filtered data set and the final orbit.

    Args:
        data_file (string): The name of the .csv file containing the positional data
        error_apriori (float): apriori estimation of the measurements error in km

    Returns:
        Runs the whole process of the program
    '''

    # First read the csv file called "orbit" with the positional data
    print("Imported file format is:", read_data.detect_file_format(data_file)["file"])
    print("")
    data = read_data.load_data(data_file)

    if(units == 'm'):
        # Transform m to km
        data[:, 1:4] = data[:, 1:4] / 1000

    print("***********Choose filter(s) in desired order of application***********")
    print("(SPACE to toggle, UP/DOWN to navigate, RIGHT/LEFT to select/deselect and ENTER to submit)")
    print("*if nothing is selected, Triple Moving Average followed by Savitzky Golay will be applied")
    questions = [
      inquirer.Checkbox('filter',
                        message="Select filter(s)",
                        choices=['None', 'Savitzky Golay Filter', 'Triple Moving Average Filter','Wiener Filter'],
                        ),
    ]
    choices = inquirer.prompt(questions)
    data_after_filter = data

    if(len(choices['filter']) == 0):
        print("Applying Triple Moving Average followed by Savitzky Golay...")
        # Apply the Triple moving average filter with window = 3
        data_after_filter = triple_moving_average.generate_filtered_data(data_after_filter, 3)

        # Use the golay_window.py script to find the window for the Savitzky Golay filter based on the error you input
        window = golay_window.window(error_apriori, data_after_filter)
        
        polyorder = 3
        if polyorder < window:
            # Apply the Savitzky Golay filter with window = window (51 for example_data/orbit.csv) and polynomial order = 3
            data_after_filter = sav_golay.golay(data_after_filter, window, polyorder)

    else:
        for index, choice in enumerate(choices['filter']):
            if(choice == 'None'):
                print("Using the original data...")
                # no filter is applied
                data_after_filter = data_after_filter

            elif (choice == 'Savitzky Golay Filter'):
                print("Applying Savitzky Golay Filter...")
                # Use the golay_window.py script to find the window for the Savitzky Golay filter
                # based on the error you input
                window = golay_window.window(error_apriori, data_after_filter)

                polyorder = 3
                if polyorder < window:
                    # Apply the Savitzky Golay filter with window = window (51 for example_data/orbit.csv) and polynomial order = 3
                    data_after_filter = sav_golay.golay(data_after_filter, window, polyorder)

            elif(choice == 'Wiener Filter'):
                print("Applying Wiener Filter...")
                # Apply the Wiener filter
                data_after_filter = wiener.wiener_new(data_after_filter, 3)
                
            else:
                print("Applying Triple Moving Average Filter...")
                # Apply the Triple moving average filter with window = 3
                data_after_filter = triple_moving_average.generate_filtered_data(data_after_filter, 3)

              
            

    # Compute the residuals between filtered data and initial data and then the sum and mean values of each axis
    res = data_after_filter[:, 1:4] - data[:, 1:4]
    sums = np.sum(res, axis = 0)
    print("\nDisplaying the sum of the residuals for each axis")
    print(sums, "\n")

    means = np.mean(res, axis = 0)
    print("Displaying the mean of the residuals for each axis")
    print(means, "\n")

    # Save the filtered data into a new csv called "filtered"
    np.savetxt("filtered.csv", data_after_filter, delimiter = ",")

    print("***********Choose Method(s) for Orbit Determination***********")
    print("(SPACE to toggle, UP/DOWN to navigate, RIGHT/LEFT to select/deselect and ENTER to submit)")
    print("*if nothing is selected, Cubic Spline Interpolation will be used for Orbit Determination")
    questions = [
      inquirer.Checkbox('method',
                        message="Select Method(s)",
                        choices=['Lamberts Kalman',
                                 'Cubic Spline Interpolation',
                                 'Ellipse Best Fit',
                                 'Gibbs 3 Vector',
                                 'Gauss 3 Vector',
                                 'MCMC (exp.)'],
                        ),
    ]
    choices = inquirer.prompt(questions)
    kep_elements = {}

    if(len(choices['method']) == 0):
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

                #previously, all data...
                #kep_lamb = lamberts_kalman.create_kep(data_after_filter)

                # only three (3) observations from half an orbit.
                # also just two (2) observations are fine for lamberts.
                data = np.array([data_after_filter[:, :][0],
                                 data_after_filter[:, :][len(data_after_filter) // 2],
                                 data_after_filter[:, :][-1]])

                kep_lamb = lamberts_kalman.create_kep(data)

                # Determination of orbit period
                semimajor_axis = kep_lamb[0][0]
                timestamps = data_after_filter[:, 0]

                index = get_timestamp_index_by_orbitperiod(semimajor_axis, timestamps)

                # enough data for half orbit
                data = np.array([data_after_filter[:, :][0],
                                 data_after_filter[:, :][index // 2],
                                 data_after_filter[:, :][index]])

                kep_lamb = lamberts_kalman.create_kep(data)


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

            elif (choice == 'Gibbs 3 Vector'):
                # Apply the Gibbs method

                # first only with first, middle and last measurement
                R = np.array([data_after_filter[:, 1:][0],
                              data_after_filter[:, 1:][len(data_after_filter) // 2],
                              data_after_filter[:, 1:][-1]])

                kep_gibbs = gibbs_method.gibbs_get_kep(R)


                # Determination of orbit period
                semimajor_axis = kep_gibbs[0][0]
                timestamps = data_after_filter[:, 0]

                index = get_timestamp_index_by_orbitperiod(semimajor_axis, timestamps)

                # enough data for half orbit
                R = np.array([data_after_filter[:, 1:][0],
                              data_after_filter[:, 1:][index // 2],
                              data_after_filter[:, 1:][index]])

                kep_gibbs = gibbs_method.gibbs_get_kep(R)

                # Apply Kalman filters to find the best approximation of the keplerian elements for all solutions
                # We set an estimate of measurement variance R = 0.01 ** 2
                kep_final_gibbs = lamberts_kalman.kalman(kep_gibbs, 0.01 ** 2)
                kep_final_gibbs = np.transpose(kep_final_gibbs)
                kep_final_gibbs = np.resize(kep_final_gibbs, ((7, 1)))
                kep_final_gibbs[6, 0] = sgp4.rev_per_day(kep_final_gibbs[0, 0])
                kep_elements['Gibbs 3 Vector'] = kep_final_gibbs

            elif (choice == 'Gauss 3 Vector'):
                # Apply the Gauss method

                # first only with first, middle and last measurement
                R = np.array([data_after_filter[:, 1:][0],
                              data_after_filter[:, 1:][len(data_after_filter) // 2],
                              data_after_filter[:, 1:][-1]])

                t1 = data_after_filter[:, 0][0]
                t2 = data_after_filter[:, 0][len(data_after_filter) // 2]
                t3 = data_after_filter[:, 0][-1]

                v2 = gauss_method.gauss_method_get_velocity(R[0], R[1], R[2], t1, t2, t3)

                # Determination of orbit period
                semimajor_axis = oe.semimajor_axis(R[0], v2)
                timestamps = data_after_filter[:, 0]

                index = get_timestamp_index_by_orbitperiod(semimajor_axis, timestamps)

                # enough data for half orbit
                R = np.array([data_after_filter[:, 1:][0],
                              data_after_filter[:, 1:][index // 2],
                              data_after_filter[:, 1:][index]])

                t1 = data_after_filter[:, 0][0]
                t2 = data_after_filter[:, 0][index // 2]
                t3 = data_after_filter[:, 0][index]

                v2 = gauss_method.gauss_method_get_velocity(R[0], R[1], R[2], t1, t2, t3)

                semimajor_axis = oe.semimajor_axis(R[0], v2)
                ecc = oe.eccentricity_v(R[1], v2)
                ecc = np.linalg.norm(ecc)
                inc = oe.inclination(R[1], v2) * 180.0 / np.pi
                AoP = oe.AoP(R[1], v2) * 180.0 / np.pi
                raan = oe.raan(R[1], v2) * 180.0 / np.pi
                true_anomaly = oe.true_anomaly(R[1], v2) * 180.0 / np.pi
                T_orbitperiod = oe.T_orbitperiod(semimajor_axis=semimajor_axis)
                n_mean_motion_perday = oe.n_mean_motion_perday(T_orbitperiod)

                kep_gauss = np.array([[semimajor_axis, ecc, inc, AoP, raan, true_anomaly, n_mean_motion_perday]])

                # Apply Kalman filters to find the best approximation of the keplerian elements for all solutions
                # We set an estimate of measurement variance R = 0.01 ** 2
                kep_final_gauss = lamberts_kalman.kalman(kep_gauss, 0.01 ** 2)
                kep_final_gauss = np.transpose(kep_final_gauss)
                kep_final_gauss = np.resize(kep_final_gauss, ((7, 1)))
                kep_final_gauss[6, 0] = sgp4.rev_per_day(kep_final_gauss[0, 0])
                kep_elements['Gauss 3 Vector'] = kep_final_gauss

            else:
                # apply mcmc method, a real optimizer

                # all data
                timestamps = data_after_filter[:, 0]
                R = np.array(data_after_filter[:, 1:])

                # all data can make the MCMC very slow. so we just pick a few in random, but in order.
                timestamps_short = []
                R_short = []
                if len(timestamps) > 25:
                    print("Too many positions for MCMC. Just 25 positons are selected")

                    # pick randomly, but in order and no duplicates
                    l = list(np.linspace(0, len(timestamps) - 1, num=len(timestamps)))
                    select_index = sorted(random.sample(list(l)[1:-1], k=23))
                    print(select_index)

                    timestamps_short.append(timestamps[0])
                    R_short.append(R[0])

                    for select in range(len(select_index)):
                        timestamps_short.append(timestamps[int(select_index[select])])
                        R_short.append(R[int(select_index[select])])

                    timestamps_short.append(timestamps[-1])
                    R_short.append(R[-1])

                else:
                    timestamps_short = timestamps
                    R_short = R


                parameters = with_mcmc.from_position(timestamps_short, R_short)

                r_a = parameters["r_a"]
                r_p = parameters["r_p"]
                AoP = parameters["AoP"]
                inc = parameters["inc"]
                raan = parameters["raan"]
                tp = parameters["tp"]

                semimajor_axis = (r_p + r_a) / 2.0
                ecc = (r_a - r_p) / (r_a + r_p)
                T_orbitperiod = oe.T_orbitperiod(semimajor_axis=semimajor_axis)
                true_anomaly = tp / T_orbitperiod * 360.0
                n_mean_motion_perday = oe.n_mean_motion_perday(T_orbitperiod)

                kep_mcmc = np.array([[semimajor_axis, ecc, inc, AoP, raan, true_anomaly, n_mean_motion_perday]])

                kep_elements['MCMC (exp.)'] = kep_mcmc


    kep_final = np.zeros((7, len(kep_elements)))
    order = []
    for index, key in enumerate(kep_elements):
        kep_final[:, index] = np.ravel(kep_elements[key])
        order.append(str(key))

    # Print the final orbital elements for all solutions
    kep_elements = ["Semi major axis (a)(km)", "Eccentricity (e)", "Inclination (i)(deg)",
                    "Argument of perigee (ω)(deg)", "Right acension of ascending node (Ω)(deg)",
                    "True anomaly (v)(deg)", "Frequency (f)(rev/day)"]

    for i in range(0, len(order)):
        print("\n******************Output for %s Method******************\n" % order[i])
        for j in range(0, 7):
            print("%s: %.16f" % (kep_elements[j], kep_final[j, i]))

    print("\nShow plots? [y/n]")
    user_input = input()

    if(user_input == "y" or user_input == "Y"):
        for j in range(0, len(order)):
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
            ax = plt.axes(projection = '3d')
            ax.plot(data[:, 1], data[:, 2], data[:, 3], ".", label = 'Initial data ')
            ax.plot(data_after_filter[:, 1], data_after_filter[:, 2], data_after_filter[:, 3], "k", linestyle = '-',
                    label = 'Filtered data')
            ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label = 'Orbit after %s method' % order[j])
            ax.legend()
            ax.can_zoom()
            ax.set_xlabel('x (km)')
            ax.set_ylabel('y (km)')
            ax.set_zlabel('z (km)')
            plt.show()
         

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type = str, help = "path to .csv data file", default = 'example_data/orbit.csv')
    parser.add_argument('-e', '--error', type = float, help = "estimation of the measurement error", default = 10.0)
    parser.add_argument('-u', '--units', type = str, help = "m for metres, k for kilometres", default = 'm')
    return parser.parse_args()


if __name__ == "__main__":
    print("\n************Welcome To OrbitDeterminator************\n")
    print("Workflow for OrbitDeterminator is as follows:")
    workflow = "                   -----------    ----------------------\n"\
               "Positional data--->| Filters |--->| Keplerian elements |--->Determined Orbit\n"\
               "                   |         |    | Determination      |\n"\
               "                   -----------    ----------------------\n\n"\
               "Available filters:               | Available methods for orbit determination:\n"\
               "  1. None (original data)        |   1. Lamberts Kalman\n"\
               "  2. Savitzky Golay Filter       |   2. Cubic spline interpolation\n"\
               "  4. Triple Moving Average Filter|   3. Ellipse Bset Fit\n"\
               "  5. Wiener Filter               |   4. Gibbs 3 Vector\n"\
               "                                 |   5. Gauss 3 Vector\n"\
               "                                 |   6. MCMC (experimental)\n"
    print("\n" + workflow)

    args = read_args()
    process(args.file_path, args.error, args.units)
    animate_orbit.animate(args.file_path, 6400)