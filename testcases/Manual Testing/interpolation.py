'''
Author: Nilesh Chaturvedi
Date Created:4th July, 2017

Description: Interpoaltion using splines for calculating velocity at a point
 and hence the orbital elements.
'''
import os
import sys
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from scipy.interpolate import CubicSpline

import state_kep
import read_data

def cubic_spline(orbit_data):
    ''' Compute component wise cubic spline of points of input data

        Args:
            orbit_data (numpy array): array of orbit data points of the
                 format [time, x, y, z]

        Returns:
            splines (list): component wise cubic splines of orbit data points
                 of the format [spline_x, spline_y, spline_z]
    '''
    time = orbit_data[:,:1]
    coordinates = list([orbit_data[:,1:2], orbit_data[:,2:3], orbit_data[:,3:4]])
    splines = list(map(lambda a:CubicSpline(time.ravel(),a.ravel()), coordinates))

    return splines

def compute_velocity(spline, point):
    ''' Calculate the velocity at a point given the spline.

    Calculate the deraivative of spline at the point(on the points the
     given spline corresponds to). This gives the velocity at that point.

    Args:
        spline (list): component wise cubic splines of orbit data points
             of the format [spline_x, spline_y, spline_z].
        point (numpy array): point at which velocity is to be calculated.

    Returns:
        velocity (numpy array): velocity vector at the given point
    '''
    velocity = list(map(lambda s, x:s(x, 1), spline, point))

    return np.array(velocity)

def main(track):
    ''' Orbital parameters for a track'''
    
    data_points = track
    #data_points = read_data.load_data(track)
    data_points[:, 1] *= 0.001
    data_points[:, 2] *= 0.001
    data_points[:, 3] *= 0.001
    velocity_vectors = []
    keplerians = []

    #for index in range(len(data_points)-1):
    for index in range(1, len(data_points)-1):
        # Take a pair of points from data_points
        spline_input = data_points[index:index+2]

        # Calculate spline corresponding to these two points
        spline = cubic_spline(spline_input)

        # Calculate velocity corresponding 1st of the 2 points of spline_input
        velocity = compute_velocity(spline, spline_input[0:,1:4][0])

        # Calculate keplerian elements correspong to the state vector(position, velocity)
        orbital_elements = state_kep.state_kep(spline_input[0:,1:4][0], velocity)

        velocity_vectors.append(velocity)
        keplerians.append(orbital_elements)

    # Uncomment the below statement to save the velocity vectors in a csv file.
    # np.savetxt('velo.csv', velocity_vectors, delimiter=",")

    # Take average of the keplerian elements corresponding to all the state vectors
    orbit = np.array(keplerians).mean(axis=0)

    return orbit 
    #[print(i) for i in velocity_vectors]
    #np.savetxt("vv.csv", velocity_vectors, delimiter=",")


if __name__ == "__main__":

    track = "track/orbit_simulated_1500453681d357_0.csv"
    path_meta = os.getcwd() + "/meta/"
    a = list(csv.reader(open(path_meta +"orbit_simulated_1500453681d357_0_meta.csv" , "r"), delimiter = "\t"))[1]
    true_val = np.array(a[4:10], dtype = np.float32)
    
    print(main(track))
    print(main(track) - true_val)
 