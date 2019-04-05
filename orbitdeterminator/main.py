'''
Runs the whole process in one file for a .csv positional data file (time, x, y, z)
and generates the final set of keplerian elements along with a plot and a filtered.csv data file
'''


from util import (read_data, kep_state, rkf78, golay_window)
from filters import (sav_golay, triple_moving_average)
from kep_determination import (lamberts_kalman, interpolation, ellipse_fit, gibbsMethod)
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import inquirer


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

    print("Choose filter(s) in the order you want to apply them")
    print("(SPACE to change, UP/DOWN to navigate, RIGHT/LEFT to Select/Deselect and ENTER to Submit)")
    print("*if nothing is selected program will run without applying filters")
    questions = [
      inquirer.Checkbox('filter',
                        message="Select filters",
                        choices=['Savintzky Golay Filter', 'Tripple Moving Average Filter'],
                        ),
    ]
    choices = inquirer.prompt(questions)
    data_after_filter = data

    if (len(choices['filter']) == 0):
        print("No filter selected, continuing without applying filter")
    else:
        for index, choice in enumerate(choices['filter']):

            if (choice == 'Savintzky Golay Filter'):
                print("Applying Savintzky Golay Filter")
                # Use the golay_window.py script to find the window for the savintzky golay filter based on the error you input
                window = golay_window.window(error_apriori, data_after_filter)

                # Apply the Savintzky - Golay filter with window = 31 and polynomail parameter = 6
                data_after_filter = sav_golay.golay(data_after_filter, window, 3)
            else:
                print("Applying Triple Moving Average Filter")
                # Apply the Triple moving average filter with window = 3
                data_after_filter = triple_moving_average.generate_filtered_data(data_after_filter, 3)


    # Compute the residuals between filtered data and initial data and then the sum and mean values of each axis
    res = data_after_filter[:, 1:4] - data[:, 1:4]
    sums = np.sum(res, axis=0)
    print("\nDisplaying the sum of the residuals for each axis")
    print(sums)

    means = np.mean(res, axis=0)
    print("Displaying the mean of the residuals for each axis")
    print(means, "\n")

    # Save the filtered data into a new csv called "filtered"
    np.savetxt("filtered.csv", data_after_filter, delimiter=",")


    print("Choose Methods for Orbit Determination")
    print("(SPACE to change, UP/DOWN to navigate, RIGHT/LEFT to Select/Deselect and ENTER to Submit)")
    print("*if nothing is selected program will determine orbit using Cubic Spline Interpolation")
    questions = [
      inquirer.Checkbox('method',
                        message="Select Methods",
                        choices=['Lamberts Kalman Solutions', 'Cubic Spline Interpolation', 'Ellipse Best Fit'],
                        ),
    ]
    choices = inquirer.prompt(questions)
    kep_elements = {}

    if (len(choices['method']) == 0):
        # Apply the interpolation method
        kep_inter = interpolation.main(data_after_filter)
        # Apply Kalman filters, estimate of measurement variance R = 0.01 ** 2
        kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)
        kep_elements['Interpolation'] = np.transpose(kep_final_inter)
    else:
        for index, choice in enumerate(choices['method']):
            if (choice == 'Lamberts Kalman Solutions'):
                # Apply Lambert's solution for the filtered data set
                kep_lamb = lamberts_kalman.create_kep(data_after_filter)
                # Apply Kalman filters, estimate of measurement variance R = 0.01 ** 2
                kep_final_lamb = lamberts_kalman.kalman(kep_lamb, 0.01 ** 2)
                kep_elements['Lambert'] = np.transpose(kep_final_lamb)

            elif (choice == 'Cubic Spline Interpolation'):
                # Apply the interpolation method
                kep_inter = interpolation.main(data_after_filter)
                # Apply Kalman filters, estimate of measurement variance R = 0.01 ** 2
                kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)
                kep_elements['Interpolation'] = np.transpose(kep_final_inter)
            elif (choice == 'Ellipse Best Fit'):
                # Fitting an ellipse on filtered data
                kep_elements['Ellipse-Fit'] = (ellipse_fit.determine_kep(data_after_filter[:, 1:]))[0]

    kep_final = np.zeros((6, len(kep_elements)))
    order = []
    for index, key in enumerate(kep_elements):
        kep_final[:, index] = np.ravel(kep_elements[key])
        order.append(str(key))

    # Print the final orbital elements for both solutions
    print("Displaying the final keplerian elements with methods in column order: {}".format(", ".join(order)))
    print(kep_final)
    print("\n")

    # Plot the initial data set, the filtered data set and the final orbit


    # First we transform the set of keplerian elements into a state vector
    state = kep_state.kep_state(kep_elements[order[0]])


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
    ax.plot(positions[0, :], positions[1, :], positions[2, :], "r-", label='Orbit after {} method'.format(order[0]))
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
    workflow = "              -----------    ------------------------------------                            \n"\
               "Input data--->| Filters |--->| Keplerian elements determination |--->Orbit of given satellite\n"\
               "              -----------    ------------------------------------                            \n\n"\
               "Available filters -                    |   Available methods for keplerian determination-\n"\
               "  1. Savintzky Golay Filter            |     1. Lamberts Kalman\n"\
               "  2. Tripple Moving Average Filter     |     2. Cubic spline interpolation\n"\
               "                                       |     3. Gibbs 3 Vector\n"\
               "                                       |     4. Ellipse Bset Fit"
    print("\nWorkflow of the program is given below in flowchart-\n" + workflow)
    print("----------------------------------------------------------------------------------------------\n\n")
    args = read_args()
    process(args.file_path, args.error, args.units)
