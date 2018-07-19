import numpy as np
import matplotlib.pyplot as plt
import gauss_method as gm
from jplephem.spk import SPK

# load JPL DE430 ephemeris SPK kernel, including TT-TDB difference
# 'de430t.bsp' may be downloaded from
# ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de430t.bsp
spk_kernel = SPK.open('de430t.bsp')
# print(spk_kernel)

# path of file of optical MPC-formatted observations
body_fname_str = '../example_data/mpc_eros_data.txt'
# load MPC data for a given NEA
mpc_object_data = gm.load_mpc_data(body_fname_str)
# print('MPC observation data:\n', mpc_object_data[ inds ], '\n')

#body name
body_name_str = 'Eros'

#load MPC data of listed observatories (longitude, parallax constants C, S) (~7,000 observations)
mpc_observatories_data = gm.load_mpc_observatories_data('mpc_observatories.txt')

#definition of the astronomical unit in km
au = 1.495978707e8

# Sun's G*m value
mu_Sun = 0.295912208285591100E-03 # au^3/day^2
mu = mu_Sun

#lines of observations file to be used for orbit determination
obs_arr = [2341,2352,2362,2369,2377,2386,2387]

#the total number of observations used
nobs = len(obs_arr)

print('nobs = ', nobs)
print('obs_arr = ', obs_arr)

#auxiliary arrays
x_vec = np.zeros((nobs-2,))
y_vec = np.zeros((nobs-2,))
z_vec = np.zeros((nobs-2,))
x_Ea_vec = np.zeros((nobs-2,))
y_Ea_vec = np.zeros((nobs-2,))
z_Ea_vec = np.zeros((nobs-2,))
a_vec = np.zeros((nobs-2,))
e_vec = np.zeros((nobs-2,))
I_vec = np.zeros((nobs-2,))
W_vec = np.zeros((nobs-2,))
w_vec = np.zeros((nobs-2,))

r2_root_ind_vec = np.zeros((nobs-2,), dtype=int)
# r2_root_ind_vec[3] = 2
r2_root_ind_vec[4] = 1
# r2_root_ind_vec[0] = 1


print('r2_root_ind_vec = ', r2_root_ind_vec)
print('len(range (0,nobs-2)) = ', len(range (0,nobs-2)))

for j in range (0,nobs-2):
    # Apply Gauss method to three elements of data
    inds_ = [obs_arr[j]-1, obs_arr[j+1]-1, obs_arr[j+2]-1]
    print('j = ', j)
    r1, r2, r3, v2, R, rho1, rho2, rho3, rho_1_, rho_2_, rho_3_, Ea_hc_pos = gm.gauss_method_mpc(spk_kernel, mpc_object_data, mpc_observatories_data, inds_, refiters=5, r2_root_ind=r2_root_ind_vec[j])

    # print('|r1| = ', np.linalg.norm(r1,ord=2))
    # print('|r2| = ', np.linalg.norm(r2,ord=2))
    # print('|r3| = ', np.linalg.norm(r3,ord=2))
    # print('r2 = ', r2)
    # print('v2 = ', v2)

    a_num = gm.semimajoraxis(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)
    e_num = gm.eccentricity(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu)

    a_vec[j] = a_num
    e_vec[j] = e_num
    I_vec[j] = np.rad2deg( gm.inclination(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
    W_vec[j] = np.rad2deg( gm.longascnode(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]) )
    w_vec[j] = np.rad2deg( gm.argperi(r2[0], r2[1], r2[2], v2[0], v2[1], v2[2], mu) )
    x_vec[j] = r2[0]
    y_vec[j] = r2[1]
    z_vec[j] = r2[2]
    x_Ea_vec[j] = Ea_hc_pos[1][0]
    y_Ea_vec[j] = Ea_hc_pos[1][1]
    z_Ea_vec[j] = Ea_hc_pos[1][2]

# print(a_num/au, 'au', ', ', e_num)
# print(a_num, 'au', ', ', e_num)
# print('j = ', j, 'obs_arr[j] = ', obs_arr[j])

# print('x_vec = ', x_vec)
# print('a_vec = ', a_vec)
# print('e_vec = ', e_vec)
print('a_vec = ', a_vec)
print('len(a_vec) = ', len(a_vec))
print('len(a_vec[a_vec>0.0]) = ', len(a_vec[a_vec>0.0]))

print('e_vec = ', e_vec)
print('len(e_vec) = ', len(e_vec))
e_vec_fil1 = e_vec[e_vec<1.0]
e_vec_fil2 = e_vec_fil1[e_vec_fil1>0.0]
print('len(e_vec[e_vec<1.0]) = ', len(e_vec_fil2))

print('*** AVERAGE ORBITAL ELEMENTS: a, e, I, Omega, omega ***')
print(np.mean(a_vec), 'au', ', ', np.mean(e_vec), ', ', np.mean(I_vec), 'deg', ', ', np.mean(W_vec), 'deg', ', ', np.mean(w_vec), 'deg')

###########################
# Plot

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
plot_lims_xyz = 1.2

# Sun-centered orbits: Computed orbit and Earth's
ax.scatter3D(x_vec[x_vec!=0.0], y_vec[x_vec!=0.0], z_vec[x_vec!=0.0], color='red', marker='.', label=body_name_str+' orbit')
ax.scatter3D(x_Ea_vec[x_Ea_vec!=0.0], y_Ea_vec[x_Ea_vec!=0.0], z_Ea_vec[x_Ea_vec!=0.0], color='blue', marker='.', label='Earth orbit')
ax.scatter3D(0.0, 0.0, 0.0, color='yellow', label='Sun')
plt.legend()
ax.set_xlabel('x (au)')
ax.set_ylabel('y (au)')
ax.set_zlabel('z (au)')
ax.set_xlim(-plot_lims_xyz, plot_lims_xyz)
ax.set_ylim(-plot_lims_xyz, plot_lims_xyz)
ax.set_zlim(-plot_lims_xyz, plot_lims_xyz)
plt.title('Angles-only orbit determ. (Gauss): '+body_name_str)
plt.show()

