import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process
import qgrid


## This code will take as input an numpy array with positional satellite data sets and will present it to the
## user using pandas and matplotilib 3d graph



## First part the numpy array holding the time,x,y,z positional data

from numpy import genfromtxt
name = 'orbit.csv'
def get_data(folder):
    my_data = genfromtxt(name, delimiter = ',')
    return my_data
my_data = get_data(name)



## present the values with pandas frame

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
def pandas_data():
    df = pd.DataFrame(my_data)
    df = df.rename(columns={0: 'Time (sec)', 1: 'x (km)', 2: 'y (km)', 3: 'z (km)'})
    return df



## present a 3d matplotlib graph diplaying the orbit

def graph():
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(my_data[:,1], my_data[:,2], my_data[:,3], "o-", label='Orbit visualization')
    ax.legend()
    ax.can_zoom()
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_zlabel('z (km)')
    plt.show()



## present a graph about the absolute value of the position vector r = (x**2 + y**2 + z**2) ** (0.5) which will help us
## identify some extreme jittery values of the data set

def absolute_value():

    r = np.zeros((len(my_data), 1))
    for i in range(0,len(my_data)):
        r[i, 0] = (my_data[i, 1]**2 + my_data[i, 2]**2 + my_data[i, 3]**2) ** (0.5)

    df2 = pd.DataFrame(r)
    df2 = df2.rename(columns={0: 'r (km)'})

    # print (df2)
    return r

def absolute_graph():
    r = absolute_value()
    fig_r = plt.figure()
    ax1 = plt.gca()


    plt.plot(my_data[:,0], r, "o-", label='Absolute value of position vector r')
    ax1.legend()
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('|r| (km)')

    plt.show()


## find some extreme jittery values by seeking big differences between two consecutive values of the |r|
## first off we find the mean value and std for all the two consecutive differences
## then we use these information to identify some extreme values and isolate them

def extreme_values():

    r = absolute_value()
    dif = np.zeros(((len(r)-1), 1))
    for i in range(0, (len(r)-1)):
        dif[i, 0] = r[i+1, 0] - r[i, 0]

    dif = np.absolute(dif)
    mean = np.mean(dif)
    std = np.std(dif)


    extreme = list()
    position = 0
    for i in range(0, len(dif)):
        if dif[i, 0] > (mean+2*std):
            extreme.append(position)
        elif dif[i, 0] < (mean-2*std):
            extreme.append(position)
        position = position + 1

    data_for_pd = np.zeros((len(extreme), 4))
    j = 0
    print('These are the extreme difference values found in the data set (position value):')
    for i in extreme:
        data_for_pd[j, :] = my_data[(i+1), :]
        j += 1

    df = pd.DataFrame(data_for_pd)
    df = df.rename(columns={0: 'Time (sec)', 1: 'x (km)', 2: 'y (km)', 3: 'z (km)'})
    print(df)

    ## delete these values from the initial data set
    extreme = [x.__add__(1) for x in extreme]

    newmy_data = np.delete(my_data, (extreme), axis = 0)








if __name__ == "__main__":

    print("Displaying the positional data set")
    df = pandas_data()
    print (df)


    while True:

        user = input('Do you want to see the graphical representation of the data you inserted? (Y/N):')
        if user == "Y":
            print('Use the left click to rotate the grafh and the right click to zoom in and out')
            work = graph()
            break
        elif user == "N":
            break
        else:
            print('Please provide a letter like Y or N')


    while True:
        user = input('Do you want to see the graphical representation of the absolute value of the positional vector (Y/N):')
        if user == "Y":
            work = absolute_graph()
            break
        elif user == "N":
            break
        else:
            print('Please provide a letter like Y or N')


    while True:
        user = input( 'Do you want to see and delete some extremely jittery data (Y/N):')
        if user == "Y":
            work = extreme_values()
            print('These data have been deleted from the initial data set')
            break
        elif user == "N":
            break
        else:
            print('Please provide a letter like Y or N')

    user = input('Press ENTER to end program')
