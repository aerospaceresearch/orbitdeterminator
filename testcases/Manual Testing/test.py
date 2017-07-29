from read_data import load_data

import numpy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file = numpy.genfromtxt("track/orbit_simulated_1500453681d357_0.csv")

#[print(float(i)) for i in file[:, :1]]
x, y, z = [], [], []
y = [x.append(float(i)) for i in file[:, 1][1:100]]
y = [y.append(float(i)) for i in file[:, 2][1:100]]
z = [z.append(float(i)) for i in file[:, 3][1:100]]
#print (x)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, "*", label="orbit")
ax.legend()
plt.show()