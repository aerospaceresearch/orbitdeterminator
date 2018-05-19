import argparse
from functools import partial

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def read_args():
    '''Reads command line arguments.

       Returns: Parsed arguments.'''

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='path to .csv file', default='orbit.csv')
    parser.add_argument('-u', '--units', type=str, help='units of distance (m or km)', default='km')
    return parser.parse_args()


def is_retro(data):
    '''Returns whether the orbit is retrograde'''

    cross_sum = 0
    for i in range(len(data)-1):
        v1 = data[i]
        v2 = data[i+1]
        cross_sum = cross_sum + np.cross(v1,v2)

    return cross_sum[2] < 0

def plane_err(data,coeffs):
    '''Calculates the total squared error of the data wrt a plane.

       The data should be a list of points. coeffs is an array of
       3 elements - the coefficients a,b,c in the plane equation
       ax+by+c = 0.

       Arguments:
       data: A numpy array of points.
       coeffs: The coefficients of the plane ax+by+c=0.

       Returns: The total squared error wrt the plane defined by ax+by+cz = 0.'''

    a,b,c = coeffs
    return np.sum((a*data[:,0]+b*data[:,1]+c*data[:,2])**2)/(a**2+b**2+c**2)


def project_to_plane(points,coeffs):
    '''Projects points onto a plane.

       Projects a list of points onto the plane ax+by+c=0,
       where a,b,c are elements of coeffs.

       Arguments:
       coeffs: The coefficients of the plane ax+by+c=0.
       points: A numpy array of points.

       Returns:
       A list of projected points.'''

    a,b,c = coeffs

    proj_mat =  [[b**2+c**2,  -a*b   ,   -a*c  ],
                 [   -a*b  ,a**2+c**2,   -b*c  ],
                 [   -a*c  ,  -b*c   ,a**2+b**2]]

    return np.matmul(points,proj_mat)/(a**2+b**2+c**2)


def conv_to_2D(points,x,y):
    '''Finds coordinates of points in a plane wrt a basis.

       Given a list of points in a plane, and a basis of the plane,
       this function returns the coordinates of those points
       wrt this basis.

       Arguments:
       points: A numpy array of points.
       x: One vector of the basis.
       y: Another vector of the basis.

       Returns:
       Coordinates of the points wrt the basis [x,y].'''

    mat = [x[0:2],y[0:2]]
    mat_inv = np.linalg.inv(mat)
    coords = np.matmul(points[:,0:2],mat_inv)

    return coords

def cart_to_pol(points):
    '''Converts a list of cartesian coordinates into polar ones.

       Arguments:
       points: The list of points in the format [x,y].

       Returns:
       A list of polar coordinates in the format [radius,angle].'''

    pol = np.empty(points.shape)
    pol[:,0] = np.sqrt(points[:,0]**2+points[:,1]**2)
    pol[:,1] = np.arctan2(points[:,1],points[:,0])

    return pol

def ellipse_err(polar_coords,params):
    '''Calculates the total squared error of the data wrt an ellipse.

       params is a 3 element array used to define an ellipse.
       It contains 3 elements a,e, and t0.

       a is the semi-major axis
       e is the eccentricity
       t0 is the angle of the major axis wrt the x-axis.

       These 3 elements define an ellipse with one focus at origin.
       Equation of the ellipse is r = a(1-e^2)/(1+ecos(t-t0))

       The function calculates r for every theta in the data.
       It then takes the square of the difference and sums it.

       Arguments:
       polar_coords: A list of polar coordinates in the format [radius,angle].
       params: The array [a,e,t0].

       Returns:
       The total squared error of the data wrt the ellipse.'''

    a,e,t0 = params
    dem = 1+e*np.cos(polar_coords[:,1]-t0)
    num = a*(1-e**2)
    r = np.divide(num,dem)
    err = np.sum((r - polar_coords[:,0])**2)
    return err


def residuals(data,params,polar_coords,basis):
    '''Calculates the residuals after fitting the ellipse.

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

       Arguments:
       data: The list of original points.
       params: The array [semi-major axis, eccentricity, argument of periapsis]
       of the fitted ellipse.
       polar_coords: The list of 2D polar coordinates of the original points after
       projecting them onto the best-fit plane.
       basis: The basis of the best-fit plane.'''

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

def read_file(file_name):
    data = np.loadtxt(file_name,skiprows=1,usecols=(1,2,3))
    
    return data

def determine_kep(data):
    retro = is_retro(data) # check whether the orbit is retrograde
    
    # try to fit a plane to the data first.
    
    # make a partial function of plane_err by supplying the data
    plane_err_data = partial(plane_err,data)
    
    # plane is defined by ax+by+cz=0.
    p0 = [0,0,1] # make an initial guess
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
    inc = math.acos(np.clip(np.dot(p,[0,0,1])/np.linalg.norm(p),-1,1))
    # lan is the angle between the lan_vec and the x axis.
    lan = math.acos(np.clip(np.dot(lan_vec,[1,0,0])/np.linalg.norm(lan_vec),-1,1))
    
    # now we try to convert the problem into a 2D problem.
    
    # project all the points onto the plane.
    proj_data = project_to_plane(data,p)
    
    # p_x and p_y are 2 orthogonal unit vectors on the plane.
    p_x,p_y = lan_vec, project_to_plane(np.cross([0,0,1],lan_vec),p)
    p_x,p_y = p_x/np.linalg.norm(p_x), p_y/np.linalg.norm(p_y)
    
    # find coordinates of the points wrt the basis [p_x,p_y].
    coords_2D = conv_to_2D(proj_data,p_x,p_y)
    
    # now try to fit an ellipse to these points.
    
    # convert them into polar coordinates
    polar_coords = cart_to_pol(coords_2D)
    
    # make an initial guess for the parametres
    r_m = np.min(polar_coords[:,0])
    r_M = np.max(polar_coords[:,0])
    a0 = (r_m+r_M)/2
    e0 = (r_M-r_m)/(r_M+r_m)
    t00 = polar_coords[np.argmin(polar_coords[:,0]),1]
    
    params0 = [a0,e0,t00] # initial guess
    # make a partial function of ellipse_err with the data
    ellipse_err_data = partial(ellipse_err,polar_coords)
    # minimize the error
    params = minimize(ellipse_err_data,params0,method='nelder-mead',options={'maxiter':1000}).x
    
    # calculate the true anomaly of the first entry in the dataset
    true_anom = (polar_coords[0][1]-params[2])%(2*math.pi)
    
    # calculation of residuals
    res = residuals(data,params,polar_coords,np.column_stack((p_x,p_y)))
    
    # handle retrograde orbits
    if retro:
        inc = math.pi - inc
        lan = (lan + math.pi)%(2*math.pi)
        params[2] = (3*math.pi - params[2])%(2*math.pi)
        true_anom = 2*math.pi - true_anom

    kep = np.empty((6,1))
    kep[0] = params[0]
    kep[1] = params[1]
    kep[2] = math.degrees(inc)
    kep[3] = math.degrees(params[2])
    kep[4] = math.degrees(lan)
    kep[5] = math.degrees(true_anom)

    return kep,res

def print_kep(kep,res,unit):
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
    ax.axis('equal')
    
    # plot
    ax.plot3D(coords_3D[0],coords_3D[1],coords_3D[2],c = 'red',label='Fitted Ellipse')
    ax.scatter3D(data[:,0],data[:,1],data[:,2],c='black',label='Initial Data')
    
    # The Pale Blue Dot
    ax.scatter3D(0,0,0,c='blue',depthshade=False,label='Earth')
    
    ax.can_zoom()
    ax.legend()
    plt.show()

if __name__ == "__main__":
    args = read_args()
    data = read_file(args.file)
    kep, res = determine_kep(data)
    print_kep(kep,res,args.units)
    plot_kep(kep,data)
