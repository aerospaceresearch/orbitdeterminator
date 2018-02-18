from util import (rkf78, kep_state)
import numpy as np
import matplotlib.pyplot as plt
import math

# kep = np.array([[7089.0], [0.0005812], [92.0239], [218.1339], [261.2200], [180.0]])
#
# state = kep_state.kep_state(kep)
#
#
# keep_state = np.zeros((6, 600))
# ti = 0.0
# tf = 100.0
# t_hold = np.zeros((600, 1))
# x = state
# h = 0.1
# tetol = 1e-08
# for i in range(0, 600):
#
# 	keep_state[:, i] = np.ravel(rkf78.rkf78(6, ti, tf, h, tetol, x))
#
# 	x = kep_state.kep_state(kep)
# 	x[:, 0] = keep_state[:, i]
#
# 	t_hold[i, 0] = tf
# 	ti = tf
# 	print(tf)
# 	tf = tf + 100
#
# positions = np.zeros((4, 600))
# positions[1:4, :] = keep_state[0:3, :]
# positions[0, :] = np.ravel(t_hold[:])
# print(positions)
# np.savetxt("orbit_clean.csv", np.transpose(positions), delimiter=",")

#
data = np.genfromtxt("orbit_clean.csv", delimiter=",")

jitter_x = 12*(np.random.random(len(data)))
jitter_y = 12*(np.random.random(len(data)))
jitter_z = 12*(np.random.random(len(data)))

new_data = np.zeros((len(data), 4))
new_data[:, 0] = data[:, 0]
for i in range(0, len(data)):
	new_data[i, 1] = data[i, 1] + jitter_x[i]
	new_data[i, 2] = data[i, 2] + jitter_y[i]
	new_data[i, 3] = data[i, 3] + jitter_z[i]

print(new_data)
np.savetxt("orbit_jittery.csv", new_data, delimiter=",")
