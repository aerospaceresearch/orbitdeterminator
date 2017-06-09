## Orbit determination

All the scripts run in python 3.4 version (we chose 3.4 because PyKEP library is not compatible for python 3.6 version)



* orbit_output.py

an example of reading and outputing positional data via user interface

Usage: needs a orbit.csv file containing all the positional data in Time, x, y, z format
Important! the file must be .csv and called "orbit"



* kep_state.py

this code transforms keplerian elements to state vectors



* state_kep.py

this code transforms state vectors to keplerian elements



* rkf78.py

this code applies a Runge Kutta numerical Integration method to a state vector and computes state vectors at other time intervals
it helps in the process of ploting the final results



* lamberts.py

uses PyKEP library to solve the preliminary orbit determination problem via Lambert's solution and gives keplerian elements 
computed from 2 positional vectors and times



* orbit_fit.py

using lamberts.py it produces all the sets of keplerian elemenets for the whole data set (orbit.csv). For example if the data set has 
200 points then this script will compute 199 sets of keplerian elements. Then it uses kalman filtering to find the final estimation
for the keplerian elements of the orbit formed by the points in the orbit.csv file



* golay_filter.py

uses Scipy library and the Savintzky - Golay filter to smooth the initial data set (orbit.csv)
Important! the window variable needs to be around len(data_set) / 4 
Example : the points of the data set are 200 then the window needs to be around 200 / 4 = 50 but it can only be an odd number so 51
The odd number restriction comes from the Scipy documentation 



* example_init.py

Runs the whole process for the orbit_jiitery.csv file we have. 
Process : Apply the Savintzky - Golay filter, then compute all the keplerian elements with Lambert's solution and finally for all these sets
of keplerian elements do a Kalman filter to find the best estimation
It plots the filtered positions, the final orbit after Kalman and the perfect orbit 
It prints the final keplerian elements set too.



* final_init.py

Runs the whole process for an orbit.csv file
Process : Apply the Savintzky - Golay filter, then compute all the keplerian elements with Lambert's solution and finally for all these sets
of keplerian elements do a Kalman filter to find the best estimation
It plots the initial positions, the after Savintzky - Golay filter positions and the final orbit after Lambert's solution and Kalman filtering 
It prints the final keplerian elements set too.

