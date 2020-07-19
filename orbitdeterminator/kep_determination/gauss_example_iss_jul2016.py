import gauss_method as gm

# path of file of optical IOD-formatted observations
# body_fname_str = '../example_data/SATOBS-BY-U-073118.txt'
filename = '../example_data/SATOBS-ML-19200716.txt'

#body name
# body_name_str = '18152 87 055A'
bodyname = 'ISS (25544)'

#lines of observations file to be used for orbit determination
obs_arr = [1, 2, 3, 4, 5, 6] # BY observations of 18152 87 055A on 2018 Jul 31st

###modify r2_root_ind_vec as necessary if adequate root of Gauss polynomial has to be selected
###if r2_root_ind_vec is not specified, then the first positive root will always be selected by default
# r2_root_ind_vec = zeros((len(obs_arr)-2,), dtype=int)
###select adequate index of Gauss polynomial root
# r2_root_ind_vec[4] = 1

obs_arr_seq = gm.gauss_method_sat_passes(filename, obs_arr, refiters=100)
print(obs_arr_seq)


index = []
radius = []
inclination = []

# sorting out different passes from one observation file
for i in range(len(obs_arr_seq)):
    print()
    index_pass, radius_pass, velocity_pass, inclination_pass, raan_pass, eccentricity_pass, AoP_pass, \
    mean_anomaly_pass, n_mean_motion_perday_pass, T_orbitperiod_pass = \
        gm.gauss_method_sat(filename, obs_arr_seq[i], refiters=100, plot=False)

    index.append(index_pass)
    radius.append(radius_pass)
    inclination.append(inclination_pass)



# checking the results.
import numpy as np
import matplotlib.pyplot as plt

for i in range(len(inclination)):
    index_pass = []
    inclination_pass = []
    for n in range(len(inclination[i])):
        index_pass.append(index[i][n][0])
        inclination_pass.append(inclination[i][n][0])
        index_pass.append(index[i][n][1])
        inclination_pass.append(inclination[i][n][1])
        index_pass.append(index[i][n][2])
        inclination_pass.append(inclination[i][n][2])

    plt.plot(index_pass, inclination_pass, "o", label=bodyname + ' orbit pass #' + str(i))

plt.grid()
plt.title('Inclination of orbit determ. (Gauss): ' + bodyname)
plt.xlabel("observation point")
plt.ylabel("Inclination [deg]")
plt.legend()
plt.show()


# plotting the coordinates
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Earth-centered orbits: satellite orbit and geocenter
ax.scatter3D(0.0, 0.0, 0.0, color='blue', label='Earth')

for i in range(len(radius)):

    x_vec = []
    y_vec = []
    z_vec = []

    for n in range(len(radius[i])):
        if n == n:#0: # for now, all the results are plotted
            x_vec.append(radius[i][n][0][0])
            y_vec.append(radius[i][n][0][1])
            z_vec.append(radius[i][n][0][2])

        x_vec.append(radius[i][n][1][0])
        y_vec.append(radius[i][n][1][1])
        z_vec.append(radius[i][n][1][2])

        if n == n:#len(radius) - 1: # for now, all the results are plotted
            x_vec.append(radius[i][n][2][0])
            y_vec.append(radius[i][n][2][1])
            z_vec.append(radius[i][n][2][2])


    ax.scatter3D(x_vec, y_vec, z_vec, marker='+', label=bodyname + ' orbit pass #' + str(i))

plt.legend()
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (km)')
xy_plot_abs_max = np.max((np.amax(np.abs(ax.get_xlim())), np.amax(np.abs(ax.get_ylim()))))
ax.set_xlim(-xy_plot_abs_max, xy_plot_abs_max)
ax.set_ylim(-xy_plot_abs_max, xy_plot_abs_max)
ax.set_zlim(-xy_plot_abs_max, xy_plot_abs_max)
ax.legend()
ax.set_title('Angles-only orbit determ. (Gauss): ' + bodyname)
plt.show()