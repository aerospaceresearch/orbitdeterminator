+++++++++
Tutorials
+++++++++

==============================
* Run the program with main.py
==============================

For the first example we will showcase how you can use the full features of the package
with main.py. Simply executing the main.py by giving the name of .csv file that contains
the positional data of the satellite, as an argument in the function process(data_file)::

    def process(data_file, error_apriori):
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

Simply input the name of the .csv file in the format of (time, x, y, z) and **tab delimiter** like the orbit.csv that is
located in the src folder and the process will run. You also need to input a apriori estimation of the measurements
errors, which in the example case is 20km per point (points every 1 second). In the case you are using your own
positional data set you need to estimate this value and input it because it is critical for the filtering process::

    run = process("orbit.csv")

.. warning::

   If the format of you data is (time, azimuth, elevation, distance) you can use the input_transf function first and be sure that the delimiter for the data file is tab delimiter since this is the one read_data supports.

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

====================================
* Run the program with automated.py
====================================

`automated.py` is another flavour of main.py that is supposed to run on a server. It keeps listening for new files in a particular directory and processes them when they arrive.  

.. note::
   All the processing invloved in this module is identical to that of main.py.

For testing purpose some files have already put in a folder named src. These are raw unprocessed files. There is another folder named dst which contains processed files along with a graph saved in the form of svg.

To execute this script, change the directory to the script's directory::

   cd orbitdeterminator/

and run the code using python3::

   python3 automated.py

and thats it. This will keep listening for new files and process them as they arrive.

.. figure:: automated_console.jpg
.. figure:: automated_graph.svg

Process
~~~~~~~

- Initialize an empty git repository in src folder
- Read the untracked files of that folder and put them in a list
- Process the files in this list and save the results(processed data and graph) to dst folder
- Stage the processed file in the src folder in order to avoid processing the same files multiple times.
- Check for any untracked files in src and apply steps 2-4 again.

=======================
* Using certain modules
=======================

In this example we are not going to use the main.py, but some of the main modules provided. First of all lets clear the
path we are going to follow which is fairly straightforward. Note that we are going to use the same orbit.csv that is
located inside the src folder and has **tab delimeter** (read_data.py reads with this delimiter).

Process
~~~~~~~
- Read the data
- Filter the data
- Compute keplerian elements for the final orbit

So first we read the data using the util/read_data.load_data function. Just input the .csv file name into the
function and it will create a numpy array with the positional data ready to be processed::

    data = read_data.load_data("orbit.csv")

.. warning::

   If the format of you data is (time, azimuth, elevation, distance) you can use the util/input_transf.spher_to_cart
   function first. And it is critical for the x, y, z to be in kilometers.

We continue by applying the Triple moving average filter::

    data_after_filter = triple_moving_average.generate_filtered_data(data, 3)

We suggest using 3 as the window size of the filter. Came to this conclusion after a lot of testing. Next we apply
the second filter to the data set which will be of a larger window size so that we can smooth the data set in
a larger scale. The optimal window size for the Savintzky - Golay filter is being computed by the function
golay_window.c(error_apriori) in which we only have to input the apriori error estimation for the initial data set
(or the measurements error)::

    error_apriori = 20.0
    c = golay_window.c(error_apriori)

    window = len(data) / c
    window = int(window)

The other 2 lines after the use of the golay_window.c(error_apriori) are needed to compute the window size for the
Savintzky - Golay filter and again for the polynomial parameter of the filter we suggest using 3::

    data_after_filter = sav_golay.golay(data_after_filter, window, 3)

At this point we have the filtered positional data set ready to be inputed into the
Lamberts - Kalman and Spline interpolation algorithms so that the final keplerian elements can be computed::

    kep_lamb = lamberts_kalman.create_kep(data_after_filter)
    kep_final_lamb = lamberts_kalman.kalman(kep_lamb, 0.01 ** 2)
    kep_inter = interpolation.main(data_after_filter)
    kep_final_inter = lamberts_kalman.kalman(kep_inter, 0.01 ** 2)

With the above 4 lines of code the final set of 6 keplerian elements is computed by the two methods.
The output format is (semi major axis (a), eccentricity (e), inclination (i), argument of perigee (ω),
right ascension of the ascending node (Ω), true anomaly (v)). So finally, in the variables kep_final_lamb and
kep_final_inter a numpy array 1x6 has the final computed keplerian elements.

.. warning::

   If the orbit you want to compute is polar (i = 90) then we suggest you to use only the interpolation method.

===========================
Using ellipse_fit method
===========================

If a lot of points are available spread over the entire orbit, then the ellipse fit method can be used for orbit 
determination. The module ``kep_determination.ellipse_fit`` has two methods - ``determine_kep`` and ``plot_kep``. 
As the name suggests, ``determine_kep`` is used to determine the orbit and ``plot_kep`` is used to plot it. 
Call ``determine_kep`` with::

    kep,res = determine_kep(data)

where *data* is a nx3 numpy array. The ellipse_fit method does not use time information at all. Hence, the 
input format is *[(x,y,z),...]*. The method results two arguments - the first output is the Keplerian 
elements while the second output is the list of residuals.

Plot the results using the ``plot_kep`` method. Call it with::

    plot_kep(kep,data)

where *kep* is the Keplerian elements we got in the last step and data is the original data. The result should 
look like this.

.. figure:: ellipse_fit.png

===========================
Using propagation modules
===========================

Cowell Method
~~~~~~~~~~~~~~

The module ``propagation.cowell`` propagates a satellite along its orbit using numerical integration. It takes 
into account the oblateness of the Earth and atmospheric drag. The module has many methods for calculating 
drag and J2 acceleration, and integrating them. However, here we will discuss only the important ones. One is 
``propagate_state`` and the other is ``time_period``. ``propagate_state`` propagates a state vector from t1 to t2.
``time_period`` finds out the nodal time period of an orbit, given a state vector. Call ``propagate_state`` like this.::

    sf = propagate_state(si,t0,tf)

where si is the state at t0 and sf is the state at tf.

.. note::
    In all propagation related discussions a state vector is the numpy array *[rx,ry,rz,vx,vy,vz]*.

Similarly to find out time period call ``time_period`` like this.::

    t = time_period(s)

DGSN Simulator
~~~~~~~~~~~~~~~

The module ``propagation.dgsn_simulator`` can be used for simulating the DGSN. Given a satellite, it propagates 
the satellite along its orbit and periodically outputs its location. The location will have some associated with 
it. Observations will also not be exactly periodic. There will be slight variations. And sometimes observations 
might not be available (for example, the satellite is out of range of the DGSN).

To use this simulator, 3 classes are used.

- The SimParams class - This is a collection of all the simulation parameters.
- The OpWriter class - This class tells the simulator what to do with the output.
- The DGSNSimulator class - This is the actual simulator class.

To start, we must choose an OpWriter class. This will tell the simulator what to do with the output. To use it, 
extend the class and override its ``write`` method. Several sample classes have been provided. For this example we 
will use the default ``print_r`` class. This just prints the output.

Now create a SimParams object. For now, only set the kep, epoch and t0.::

    epoch = 1531152114
    t0 = epoch
    iss_kep = np.array([6785.6420,0.0003456,51.6418,290.0933,266.6543,212.4306])
    
    params = SimParams()
    params.kep = iss_kep
    params.epoch = epoch
    params.t0 = t0

Now initialize the simulator with these parameters and start it.::

    s = DGSNSimulator(params)
    s.simulate()

The program should start printing the time and the corresponding satellite coordinates on the terminal.

.. note::

    The module ``propagation.simulator`` is similar to this module. The only difference is that it doesn't 
    add any noise. So it can be used for comparison purposes.
