'''
Runs the whole process in one file for a .csv positional data file (time, x, y, z)
and generates the final set of keplerian elements along with a plot and a filtered.csv data file
'''


from util import (read_data, kep_state, rkf78, golay_window)
from filters import (sav_golay, triple_moving_average)
from kep_determination import (lamberts_kalman, interpolation, gibbsMethod, ellipse_fit)
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from propagation import sgp4

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
    data = read_data.load_data(data_file)

    if (units == 'm'):
        # Transform m to km
        data[:, 1:4] = data[:, 1:4] / 1000

    # Apply the Triple moving average filter with window = 3
    data_after_filter = triple_moving_average.generate_filtered_data(data, 3)

    ## Use the golay_window.py script to find the window for the savintzky golay filter based on the error you input
    window = golay_window.window(error_apriori, data_after_filter)
    
    # Apply the Savintzky - Golay filter with window = window (51 for orbit.csv) and polynomial order = 3
    data_after_filter = sav_golay.golay(data_after_filter, window, 3)

    # Compute the residuals between filtered data and initial data and then the sum and mean values of each axis
    res = data_after_filter[:, 1:4] - data[:, 1:4]
    sums = np.sum(res, axis=0)
    print("Displaying the sum of the residuals for each axis")
    print(sums)
    print("\n")

    means = np.mean(res, axis=0)
    print("Displaying the mean of the residuals for each axis")
    print(means)
    print("\n")

    # Save the filtered data into a new csv called "filtered"
    np.savetxt("filtered.csv", data_after_filter, delimiter=",")

    # Apply Lambert's solution for the filtered data set
    kep_lamb = lamberts_kalman.create_kep(data_after_filter)

    # Apply the interpolation method
    kep_inter = interpolation.main(data_after_filter)

    # Apply the ellipse best fit method
    kep_ellip = ellipse_fit.determine_kep(data_after_filter[:,1:])[0]

    # Apply the Gibbs method
    kep_gibbs = gibbsMethod.gibbs_get_kep(data_after_filter[:,1:])

    # Apply Kalman filters to find the best approximation of the keplerian elements for all solutions
    # We set an estimate of measurement variance R = 0.01 ** 2
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

    # Confirmation required
    # kep_final_lamb[5, 0] = kep_final_inter[5, 0] = kep_final_ellip[5, 0] = kep_final_gibbs[5, 0]

    kep_final = np.zeros((7, 4))
    kep_final[:, 0] = np.ravel(kep_final_lamb)
    kep_final[:, 1] = np.ravel(kep_final_inter)
    kep_final[:, 2] = np.ravel(kep_final_ellip)
    kep_final[:, 3] = np.ravel(kep_final_gibbs)

    # Print the final orbital elements for all solutions
    kep_elements = ["Semi major axis (a)(km)", "Eccentricity (e)", "Inclination (i)(deg)", "Argument of perigee (ω)(deg)", "Right acension of ascending node (Ω)(deg)", "True anomaly (v)(deg)", "Frequency (f)(rev/day)"]
    det_methods = ["Lamberts Kalman", "Spline Interpolation", "Ellipse Best Fit", "Gibbs 3 Vector"]

    # Not doing Gibbs for now, some error in filtering
    for i in range(0, 4):
        print("\n******************Output for %s Method******************\n" % det_methods[i])
        for j in range(0, 7):
            print("%s: %.16f" % (kep_elements[j], kep_final[j, i]))

    print("\nShow plots? [y/n]")
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
            plt.show()

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, help="path to .csv data file", default='orbit.csv')
    parser.add_argument('-e', '--error', type=float, help="estimation of the measurement error", default=10.0)
    parser.add_argument('-u', '--units', type=str, help="m for metres, k for kilometres", default='k')
    return parser.parse_args()


if __name__ == "__main__":

    args = read_args()
    process(args.file_path, args.error, args.units)
