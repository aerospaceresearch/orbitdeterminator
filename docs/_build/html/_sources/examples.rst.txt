Examples
========

Run the program with main.py
----------------------------

For the first example we will showcase how you can use the full features of the package
with main.py. Simply executing the main.py by giving the name of .csv file that contains
the positional data of the satellite, as an argument in the function process(data_file)::

def process(data_file):
    '''
    Given a .csv data file in the format of (time, x, y, z) applies both filters, generates a filtered.csv data
    file, prints out the final keplerian elements computed from both Lamberts and Interpolation and finally plots
    the initial, filtered data set and the final orbit.

    Args:
        data_file (string): The name of the .csv file containing the positional data

    Returns:
        Runs the whole process of the program
    '''

Simply input the name of the .csv file in the format of (time, x, y, z) like the orbit.csv that is located
in the example data folder inside the orbitdeterminator package and the process will run::

run = process("orbit.csv")

.. warning::

   If the format of you data is (time, azimuth, elevation, distance) you can must the input_transf function first

The process that will run with the use of the process function is, first the program reads your data from the .csv file
then, applies both filters (Triple moving average and Savintzky - Golay), generates a .csv file called filtered, that included the filtered data set,
computes the keplerian elements of the orbit with both methods (Lamberts - Kalman and Spline Interpolation) and finally prints and plots some results.
More specifically, the results printed by this process will be first the sum and mean value of the residuals
(difference between filtered and initial data), the computed keplerian elements in format of (a - semi major axis,
e - eccentricity, i - inclination, ω - argument of perigee, Ω - right ascension of the ascending node,
v - true anomaly) and a 3d matplotlib graph that plots the initial, filtered data set and the final computed orbit
described by the keplerian elements (via the interpolation method).

Process
~~~~~~~

- Reads the data
- Uses both filters on them (Triple moving average and Savintzky - Golay )
- Generates a .csv file called filtered that includes the filtered data set
- Computes keplerian elements with both methods (Lamberts - Kalman and Spline Interpolation)
- Prints results and plot a 3d matplotlib graph

Results
~~~~~~~

- Sum and mean of the residuals (differences between filtered and initial data set)
- Final keplerian elements from both methods (first column : Lamberts - Kalman, second column : Spline Interpolation)
- 3d matplotlib graph with the initial, filtered data set and the final orbit described by the keplerian elements from Spline Interpolation

.. warning::

   Measurement unit for distance is kilometer and for angle degrees

The output should look like the following image.

.. figure::  results.jpg


