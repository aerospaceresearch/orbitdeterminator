# Examples 5.11 and 5.12 from book

import numpy as np
import gauss_method as gm
from least_squares import xyz_frame_
import matplotlib.pyplot as plt

phi_deg = 40.0 # deg
altitude_km = 1.0 # km
f = 0.003353
ra_deg = np.array((43.537, 54.420, 64.318))
ra_hrs = ra_deg/15.0
dec_deg = np.array((-8.7833, -12.074, -15.105))
lst_deg = np.array((44.506, 45.000, 45.499))
t_sec = np.array((0.0, 118.10, 237.58))

# print('r2 = ', r2)
# print('v2 = ', v2)

# for i in range(0,6):
#     # print('i = ', i)
    

r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_ = gm.gauss_method_sat(phi_deg, altitude_km, f, ra_hrs, dec_deg, lst_deg, t_sec, refiters=10)

print('r2 = ', r2)
print('v2 = ', v2)

mu_Earth = 398600.435436 # Earth's G*m, km^3/seg^2
mu = mu_Earth

a = gm.semimajoraxis(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
e = gm.eccentricity(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
I = np.rad2deg( gm.inclination(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
W = np.rad2deg( gm.longascnode(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
w = np.rad2deg( gm.argperi(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu) )
theta = np.rad2deg( gm.trueanomaly(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu) )

print('a = ', a)
print('e = ', e)
print('I = ', I, 'deg')
print('W = ', W, 'deg')
print('w = ', w, 'deg')
print('theta = ', theta, 'deg')

npoints = 1000
theta_vec = np.linspace(0.0, 2.0*np.pi, npoints)
x_orb_vec = np.zeros((npoints,))
y_orb_vec = np.zeros((npoints,))
z_orb_vec = np.zeros((npoints,))

for i in range(0,npoints):
    recovered_xyz = xyz_frame_(a, e, theta_vec[i], np.deg2rad(w), np.deg2rad(I), np.deg2rad(W))
    x_orb_vec[i] = recovered_xyz[0]
    y_orb_vec[i] = recovered_xyz[1]
    z_orb_vec[i] = recovered_xyz[2]

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

xline1 = np.array((0.0, R[0][0]))
yline1 = np.array((0.0, R[0][1]))
zline1 = np.array((0.0, R[0][2]))
xline2 = np.array((0.0, R[1][0]))
yline2 = np.array((0.0, R[1][1]))
zline2 = np.array((0.0, R[1][2]))
xline3 = np.array((0.0, R[2][0]))
yline3 = np.array((0.0, R[2][1]))
zline3 = np.array((0.0, R[2][2]))
xline4 = np.array((0.0, r1[0]))
yline4 = np.array((0.0, r1[1]))
zline4 = np.array((0.0, r1[2]))
xline5 = np.array((R[0][0], R[0][0]+rho_1_*rho1[0]))
yline5 = np.array((R[0][1], R[0][1]+rho_1_*rho1[1]))
zline5 = np.array((R[0][2], R[0][2]+rho_1_*rho1[2]))
xline6 = np.array((0.0, r2[0]))
yline6 = np.array((0.0, r2[1]))
zline6 = np.array((0.0, r2[2]))
xline7 = np.array((R[1][0], R[1][0]+rho_2_*rho2[0]))
yline7 = np.array((R[1][1], R[1][1]+rho_2_*rho2[1]))
zline7 = np.array((R[1][2], R[1][2]+rho_2_*rho2[2]))
xline8 = np.array((0.0, r3[0]))
yline8 = np.array((0.0, r3[1]))
zline8 = np.array((0.0, r3[2]))
xline9 = np.array((R[2][0], R[2][0]+rho_3_*rho3[0]))
yline9 = np.array((R[2][1], R[2][1]+rho_3_*rho3[1]))
zline9 = np.array((R[2][2], R[2][2]+rho_3_*rho3[2]))
ax.plot3D(xline1, yline1, zline1, 'gray', label='Observer 1')
ax.plot3D(xline2, yline2, zline2, 'blue', label='Observer 2')
ax.plot3D(xline3, yline3, zline3, 'green', label='Observer 3')
ax.plot3D(xline4, yline4, zline4, 'orange')
ax.plot3D(xline5, yline5, zline5, 'red', label='LOS 1')
ax.plot3D(xline6, yline6, zline6, 'black')
ax.plot3D(xline7, yline7, zline7, 'cyan', label='LOS 2')
ax.plot3D(xline8, yline8, zline8, 'brown')
ax.plot3D(xline9, yline9, zline9, 'yellow', label='LOS 3')
ax.scatter3D(0.0, 0.0, 0.0, color='blue', label='Geocenter')
ax.plot3D(x_orb_vec, y_orb_vec, z_orb_vec, 'black', label='Satellite orbit')
# ax.plot_surface(x_ea_surf, y_ea_surf, z_ea_surf, color='b')
# ax.set_aspect('equal')
plt.legend()
ax.set_xlim(-10000.0, 10000.0)
ax.set_ylim(-10000.0, 10000.0)
ax.set_zlim(-10000.0, 10000.0)
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
plt.title('Satellite orbit determination: Gauss method')
plt.show()
