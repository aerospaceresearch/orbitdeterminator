"""Finds out the ellipse that best fits to a set of data points and calculates
   its keplerian elements.
"""

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from functools import partial

def __read_args():
    """Reads command line arguments.
       Returns:
           object: Parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='path to .csv file', default='orbit.csv')
    parser.add_argument('-u', '--units', type=str, help='units of distance (m or km)', default='km')
    return parser.parse_args()


def __cross_sum(data):
    """Returns the normalized sum of the cross products between consecutive vectors.
    Args:
        data(nx3 numpy array): A matrix where each column represents the x,y,z coordinates of each position vector.
    Returns:
        float: The normalized sum of the cross products between consecutive vectors.
    """

    cross_sum = 0
    for i in range(len(data)-1):
        v1 = data[i]
        v2 = data[i+1]
        cross_sum = cross_sum + np.cross(v1,v2)

    return cross_sum/np.linalg.norm(cross_sum)


def __plane_err(data,coeffs):
    """Calculates the total squared error of the data wrt a plane.
       The data should be a list of points. coeffs is an array of
       3 elements - the coefficients a,b,c in the plane equation
       ax+by+c = 0.
       Args:
           data(nx3 numpy array): A numpy array of points.
           coeffs(1x3 array): The coefficients of the plane ax+by+c=0.
       Returns:
           float: The total squared error wrt the plane defined by ax+by+cz = 0.
    """

    a,b,c = coeffs
    return np.sum((a*data[:,0]+b*data[:,1]+c*data[:,2])**2)/(a**2+b**2+c**2)


def __project_to_plane(points,coeffs):
    """Projects points onto a plane.
       Projects a list of points onto the plane ax+by+c=0,
       where a,b,c are elements of coeffs.
       Args:
           points(nx3 numpy array): A numpy array of points.
           coeffs(1x3 array): The coefficients of the plane ax+by+c=0.
       Returns:
           nx3 numpy array: A list of projected points.
    """

    a,b,c = coeffs

    proj_mat =  [[b**2+c**2,  -a*b   ,   -a*c  ],
                 [   -a*b  ,a**2+c**2,   -b*c  ],
                 [   -a*c  ,  -b*c   ,a**2+b**2]]

    return np.matmul(points,proj_mat)/(a**2+b**2+c**2)


def __conv_to_2D(points,x,y):
    """Finds coordinates of points in a plane wrt a basis.
       Given a list of points in a plane, and a basis of the plane,
       this function returns the coordinates of those points
       wrt this basis.
       Args:
           points(numpy array): A numpy array of points.
           x(3x1 numpy array): One vector of the basis.
           y(3x1 numpy array): Another vector of the basis.
       Returns:
           nx2 numpy array: Coordinates of the points wrt the basis [x,y].
    """

    mat = [x[0:2],y[0:2]]
    mat_inv = np.linalg.inv(mat)
    coords = np.matmul(points[:,0:2],mat_inv)

    return coords

def __cart_to_pol(points):
    """Converts a list of cartesian coordinates into polar ones.
       Args:
           points(nx2 numpy array): The list of points in the format [x,y].
       Returns:
           nx2 numpy array: A list of polar coordinates in the format [radius,angle].
    """

    pol = np.empty(points.shape)
    pol[:,0] = np.sqrt(points[:,0]**2+points[:,1]**2)
    pol[:,1] = np.arctan2(points[:,1],points[:,0])

    return pol

def __ellipse_err(polar_coords,params):
    """Calculates the total squared error of the data wrt an ellipse.
       params is a 3 element array used to define an ellipse.
       It contains 3 elements a,e, and t0.
       a is the semi-major axis
       e is the eccentricity
       t0 is the angle of the major axis wrt the x-axis.
       These 3 elements define an ellipse with one focus at origin.
       Equation of the ellipse is r = a(1-e^2)/(1+ecos(t-t0))
       The function calculates r for every theta in the data.
       It then takes the square of the difference and sums it.
       Args:
           polar_coords(nx2 numpy array): A list of polar coordinates in the format [radius,angle].
           params(1x3 numpy array): The array [a,e,t0].
       Returns:
           float: The total squared error of the data wrt the ellipse.
    """

    a,e,t0 = params
    dem = 1+e*np.cos(polar_coords[:,1]-t0)
    num = a*(1-e**2)
    r = np.divide(num,dem)
    err = np.sum((r - polar_coords[:,0])**2)
    return err


def __residuals(data,params,polar_coords,basis):
    """Calculates the residuals after fitting the ellipse.
       Residuals are the difference between the fitted points and
       the actual points.
       res_x = fitted_x - initial_x
       res_y = fitted_y - initial_y
       res_z = fitted_z - initial_z
       where fitted_x,y,z is the closest point on the ellipse to initial_x,y,z.
       However, it is computationally expensive to find the true nearest point.
       So we take an approximation. We consider the point on the ellipse with
       the same true anomaly as the initial point to be the nearest point to it.
       Since the eccentricities of the orbits involved are small, this approximation
       holds.
       Args:
           data(nx3 numpy array): The list of original points.
           params(1x3 numpy array): The array [semi-major axis, eccentricity, argument of periapsis]
                                    of the fitted ellipse.
           polar_coords(nx2 numpy array): The list of 2D polar coordinates of the original points after
                                          projecting them onto the best-fit plane.
           basis(3x2 numpy array): The basis of the best-fit plane.
       Returns:
            nx3 numpy array: Returns the residuals
    """

    a,e,t0 = params
    dem = 1+e*np.cos(polar_coords[:,1]-t0)
    num = a*(1-e**2)
    r = np.divide(num,dem)

    # convert to cartesian
    x_s = np.multiply(r,np.cos(polar_coords[:,1]))
    y_s = np.multiply(r,np.sin(polar_coords[:,1]))

    # convert to 3D
    filtered_coords = np.transpose(np.matmul(basis,[x_s,y_s]))

    residuals = filtered_coords - data

    return residuals

def __read_file(file_name):
    """Reads a space separated csv file with 4 columns in the format t x y z.
       Args:
           file_name(string): the path to the file
       Returns:
           nx3 numpy array: A numpy array with the columns [x y z]. Note that the t coloumn is discarded.
    """

    data = np.loadtxt(file_name,skiprows=1,usecols=(1,2,3))

    return data

def determine_kep(data):
    """Determines keplerian elements that fit a set of points.
       Args:
           data(nx3 numpy array): A numpy array of points in the format [x y z].
       Returns:
           (kep,res) - The keplerian elements and the residuals as a tuple.
           kep: 1x6 numpy array
           res: nx3 numpy array
           For the keplerian elements:
           kep[0] - semi-major axis (in whatever units the data was provided in)
           kep[1] - eccentricity
           kep[2] - inclination (in degrees)
           kep[3] - argument of periapsis (in degrees)
           kep[4] - right ascension of ascending node (in degrees)
           kep[5] - true anomaly of the first row in the data (in degrees)
           For the residuals: (in whatever units the data was provided in)
           res[0] - residuals in x axis
           res[1] - residuals in y axis
           res[2] - residuals in z axis
    """

    # try to fit a plane to the data first.

    # make a partial function of plane_err by supplying the data
    plane_err_data = partial(__plane_err,data)

    # plane is defined by ax+by+cz=0.
    p0 = __cross_sum(data) # make an initial guess

    # minimize the error
    p = minimize(plane_err_data,p0,method='nelder-mead',options={'maxiter':1000}).x
    p = p/np.linalg.norm(p) # normalize p

    # now p is the normal vector of the best-fit plane.

    # lan_vec is a vector along the line of intersection of the plane
    # and the x-y plane.
    lan_vec = np.cross([0,0,1],p)

    # if lan_vec is [0,0,0] it means that it is undefined and can take on
    # any value. So we set it to [1,0,0] so that the rest of the
    # calculation can proceed.
    if (np.array_equal(lan_vec,[0,0,0])):
        lan_vec = [1,0,0]

    # inclination is the angle between p and the z axis.
    inc = math.acos(np.clip(p[2]/np.linalg.norm(p),-1,1))
    # lan is the angle between the lan_vec and the x axis.
    lan = math.atan2(lan_vec[1],lan_vec[0])%(2*math.pi)

    # now we try to convert the problem into a 2D problem.

    # project all the points onto the plane.
    proj_data = __project_to_plane(data,p)

    # p_x and p_y are 2 orthogonal unit vectors on the plane.
    p_x,p_y = lan_vec, np.cross(p,lan_vec)
    p_x,p_y = p_x/np.linalg.norm(p_x), p_y/np.linalg.norm(p_y)

    # find coordinates of the points wrt the basis [p_x,p_y].
    coords_2D = __conv_to_2D(proj_data,p_x,p_y)

    # now try to fit an ellipse to these points.

    # convert them into polar coordinates
    polar_coords = __cart_to_pol(coords_2D)

    # make an initial guess for the parametres
    r_m = np.min(polar_coords[:,0])
    r_M = np.max(polar_coords[:,0])
    a0 = (r_m+r_M)/2
    e0 = (r_M-r_m)/(r_M+r_m)
    t00 = polar_coords[np.argmin(polar_coords[:,0]),1]

    params0 = [a0,e0,t00] # initial guess
    # make a partial function of ellipse_err with the data
    ellipse_err_data = partial(__ellipse_err,polar_coords)
    # minimize the error
    params = minimize(ellipse_err_data,params0,method='nelder-mead',options={'maxiter':1000}).x
    params[2] = params[2]%(2*math.pi)  # bring argp between 0-360 degrees

    # calculate the true anomaly of the first entry in the dataset
    true_anom = (polar_coords[0][1]-params[2])%(2*math.pi)

    # calculation of residuals
    res = __residuals(data,params,polar_coords,np.column_stack((p_x,p_y)))

    kep = np.empty((6,1))
    kep[0] = params[0]
    kep[1] = params[1]
    kep[2] = math.degrees(inc)
    kep[3] = math.degrees(params[2])
    kep[4] = math.degrees(lan)
    kep[5] = math.degrees(true_anom)

    return kep,res

def __print_kep(kep,res,unit):
    """Prints the keplerian elements and some information on residuals.
       Args:
           kep(1x6 numpy array): keplerian elements
           res(nx3 numpy array): residuals
           unit(string): units of distance used
       Returns:
           NIL
    """

    # output the parameters
    print("Semi-major axis:            ",kep[0][0],unit)
    print("Eccentricity:               ",kep[1][0])
    print("Inclination:                ",kep[2][0],"deg")
    print("Argument of periapsis:      ",kep[3][0],"deg")
    print("Longitude of Ascending Node:",kep[4][0],"deg")
    print("True Anomaly                ",kep[5][0],"deg")

    # print data about residuals
    print()

    max_res = np.max(res,axis=0)
    min_res = np.min(res,axis=0)
    sum_res = np.sum(res,axis=0)
    avg_res = np.average(res,axis=0)
    std_res = np.std(res,axis=0)

    print("Printing data about residuals in each axis:")
    print("Max:               ",max_res)
    print("Min:               ",min_res)
    print("Sum:               ",sum_res)
    print("Average:           ",avg_res)
    print("Standard Deviation:",std_res)

def plot_kep(kep,data):
    """Plots the original data and the orbit defined by the keplerian elements.
       Args:
           kep(1x6 numpy array): keplerian elements
           data(nx3 numpy array): original data
       Returns:
           nothing
    """

    a = kep[0]
    e = kep[1]
    inc = math.radians(kep[2])
    t0 = math.radians(kep[3])
    lan = math.radians(kep[4])

    p_x = np.array([math.cos(lan), math.sin(lan), 0])
    p_y = np.array([-math.sin(lan)*math.cos(inc), math.cos(lan)*math.cos(inc), math.sin(inc)])

    # generate 1000 points on the ellipse
    theta = np.linspace(0,2*math.pi,1000)
    radii = a*(1-e**2)/(1+e*np.cos(theta-t0))

    # convert to cartesian
    x_s = np.multiply(radii,np.cos(theta))
    y_s = np.multiply(radii,np.sin(theta))

    # convert to 3D
    mat = np.column_stack((p_x,p_y))
    coords_3D = np.matmul(mat,[x_s,y_s])

    fig = plt.figure()
    ax = Axes3D(fig)
    try:
      ax.set_aspect('equal')
    except NotImplementedError:
      print("Warning: The equal aspect ratio of 3d plots in matplotlib is currently an issue within the library.")
      (__x_min, __x_max) = (data[:,0].min(), data[:,0].max())
      (__y_min, __y_max) = (data[:,1].min(), data[:,1].max())
      (__z_min, __z_max) = (data[:,2].min(), data[:,2].max())
      __overall_min = min(__x_min, __y_min, __z_min)
      __overall_max = max(__x_max, __y_max, __z_max)
      ax.set_xlim(__overall_min, __overall_max)
      ax.set_ylim(__overall_min, __overall_max)
      ax.set_zlim(__overall_min, __overall_max)

    # plot
    ax.plot3D(coords_3D[0],coords_3D[1],coords_3D[2],c = 'red',label='Fitted Ellipse')
    ax.scatter3D(data[:,0],data[:,1],data[:,2],c='black',label='Initial Data')

    # The Pale Blue Dot
    ax.scatter3D(0,0,0,c='blue',depthshade=False,label='Earth')

    ax.can_zoom()
    ax.legend()
    plt.show()

if __name__ == "__main__":
    args = __read_args()
    data = __read_file(args.file)
    kep, res = determine_kep(data)
    __print_kep(kep,res,args.units)
    plot_kep(kep,data)
