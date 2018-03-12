import argparse
from functools import partial

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='path to .csv file', default='orbit.csv')
    parser.add_argument('-u', '--units', type=str, help='units of distance (m or km)', default='km')
    return parser.parse_args()

def plane_err(data,coeffs):
    a,b,c = coeffs
    return np.sum((a*data[:,0]+b*data[:,1]+c*data[:,2])**2)/(a**2+b**2+c**2)

def project_to_plane(coeffs,points):
    a,b,c = coeffs

    proj_mat =  [[b**2+c**2,  -a*b   ,   -a*c  ],
                 [   -a*b  ,a**2+c**2,   -b*c  ],
                 [   -a*c  ,  -b*c   ,a**2+b**2]]

    return np.matmul(points,proj_mat)/(a**2+b**2+c**2)

def conv_to_2D(points,x,y):
    mat = [x[0:2],y[0:2]]
    mat_inv = np.linalg.inv(mat)
    coords = np.matmul(points[:,0:2],mat_inv)

    return coords

def cart_to_pol(points):
    pol = np.empty(points.shape)
    pol[:,0] = np.sqrt(points[:,0]**2+points[:,1]**2)
    pol[:,1] = np.arctan2(points[:,1],points[:,0])#*57.296

    return pol

def ellipse_err(polar_coords,params):
    a,e,t0 = params
    dem = 1+e*np.cos(polar_coords[:,1]-t0)
    num = a*(1-e**2)
    r = np.divide(num,dem)
    err = np.sum((r - polar_coords[:,0])**2)
    return err

args = read_args()
data = np.loadtxt(args.file,skiprows=1,usecols=(1,2,3));
plane_err_data = partial(plane_err,data)

# ax+by+cz=0
p0 = [0,0,1]
p = minimize(plane_err_data,p0,method='nelder-mead').x
p = p/np.linalg.norm(p) # normalize p

lan_vec = np.cross([0,0,1],p)
inc = math.acos(np.dot(p,[0,0,1])/np.linalg.norm(p))
lan = math.acos(np.dot(lan_vec,[1,0,0])/np.linalg.norm(lan_vec))

proj_data = project_to_plane(p,data)

p_x,p_y = lan_vec, project_to_plane(p,np.cross([0,0,1],lan_vec))
p_x,p_y = p_x/np.linalg.norm(p_x), p_y/np.linalg.norm(p_y)

coords_2D = conv_to_2D(proj_data,p_x,p_y)

polar_coords = cart_to_pol(coords_2D)

r_m = np.min(polar_coords[:,0])
r_M = np.max(polar_coords[:,0])
a0 = (r_m+r_M)/2
e0 = (r_M-r_m)/(r_M+r_m)
t0 = polar_coords[np.argmin(polar_coords[:,0]),1]

params0 = [a0,e0,t0]
ellipse_err_data = partial(ellipse_err,polar_coords)
params = minimize(ellipse_err_data,params0,method='nelder-mead').x

# output
print("Semi-major axis:            ",params[0],args.units)
print("Eccentricity:               ",params[1])
print("Argument of periapsis:      ",params[2],"rad")
print("Inclination:                ",inc,"rad")
print("Longitude of Ascending Node:",lan,"rad")

# plotting
a,e,t0 = params

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

ax.plot3D(coords_3D[0],coords_3D[1],coords_3D[2],'red',label='Fitted Ellipse')
ax.scatter3D(data[::8,0],data[::8,1],data[::8,2],c='black',depthshade=False,label='Initial Data')

# Earth
ax.scatter3D(0,0,0,c='blue',s=500,depthshade=False)

ax.can_zoom()
ax.legend()
plt.show()
