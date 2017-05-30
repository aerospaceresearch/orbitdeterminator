'''
Created by Alexandros Kazantzidis
Date 25/05/17 (The basic statistical filtering was implemented in 26/05/17)
'''

import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D




## This code will take as input an numpy array with positional satellite data sets and will present it to the
## user using pandas and matplotilib 3d graph and do some simple statistical filtering 
## IMPORTANT TO RUN : it needs a csv file in the same folder called 'orbit' to work properly, 
## needs implementation with Nilesh read_data code to be completed 




## First part the numpy array holding the time,x,y,z positional data


name = 'orbit.csv'

def get_data(folder):
    my_data = genfromtxt(name, delimiter = ',')
    return my_data
my_data = get_data(name)





pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
def pandas_data():
    '''
    present the values with pandas frame
    '''

    df = pd.DataFrame(my_data)
    df = df.rename(columns={0: 'Time (sec)', 1: 'x (km)', 2: 'y (km)', 3: 'z (km)'})
    return df





def graph():
    '''
    present a 3d matplotlib graph diplaying the orbit
    '''

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

    return ax





def absolute_value():
    '''
    computes the absolute value of the position vector 
    r = (x**2 + y**2 + z**2) ** (0.5) which will help us
    identify some extreme jittery values of the data set
    '''



    r = np.zeros((len(my_data), 1))
    for i in range(0,len(my_data)):
        r[i, 0] = (my_data[i, 1]**2 + my_data[i, 2]**2 + my_data[i, 3]**2) ** (0.5)

    df2 = pd.DataFrame(r)
    df2 = df2.rename(columns={0: 'r (km)'})

    # print (df2)
    return r

def absolute_graph():
    ''' 
    plots the graph of the absolute value of the position vector r
    '''


    r = absolute_value()
    fig_r = plt.figure()
    ax1 = plt.gca()


    plt.plot(my_data[:,0], r, "o-", label='Absolute value of position vector r')
    ax1.legend()
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('|r| (km)')

    plt.show()
    return ax1




def extreme_values():
    '''
    find some extreme jittery values by seeking big differences between two consecutive values of the |r|
    first off we find the mean value and std for all the two consecutive differences
    then we use these information to identify some extreme values and isolate them
    
    Output
    
    df = a pandas dataframe containing all the jittery data that are going to be deleted
    newmy_data = the new data without the jittery ones in the form that have been inputed 
    (numpy array holding the time,x,y,z positional data)
    '''

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


    ## delete these values from the initial data set
    extreme = [x.__add__(1) for x in extreme]

    newmy_data = np.delete(my_data, (extreme), axis = 0)
    return df, newmy_data







if __name__ == "__main__":

    print("Displaying the positional data set")
    df = pandas_data()
    print(df)


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
            df, new_data = extreme_values()
            print(df)
            print('These data have been deleted from the initial data set')
            break
        elif user == "N":
            break
        else:
            print('Please provide a letter like Y or N')

    user = input('Press ENTER to end program')


