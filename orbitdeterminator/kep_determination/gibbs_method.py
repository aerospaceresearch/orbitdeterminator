'''
Source: Algorithm 5.1 on Gibb's Method in book "Orbital Mechanics for Engineering Student"

graphics.py is taken from https://github.com/Elucidation/OrbitalElements

Important:
In order to get the plot you have to first install Tkinter.
Type the following command into your terminal.

$ sudo apt-get install python-tk
'''

import urllib2
import graphics
import numpy as np

sqrt = np.sqrt
pi = np.pi

'''
class for Gibb's implementation
'''
class Gibbs:
    'Implementing Gibb\'s Method to plot orbit of a satellite'
    meu = 398600.4418

    def __init__(self):
        '''
        Initializing class variables

        Args:
            self : class variables
            filename : name of data file

        Returns:
            NIL
        '''

        self.file_len = 0
        self.axis = 0.0
        self.inc = 0.0
        self.asc = 0.0
        self.ecc = 0.0
        self.per = 0.0
        self.anom = 0.0

    def read_file(self, d):
        '''
        Read the given position vector file

        Args:
            self : class variables

        Returns:
            NIL
        '''

        self.file_len = len(d)
        for i in range(0,self.file_len-2):
            vec1 = d[i]
            vec2 = d[i+1]
            vec3 = d[i+2]
            self.gibbs(vec1, vec2, vec3)                                            # call gibb's method function

        self.average()                                                              # taking average of all the orbital elements from file
        self.display()                                                              # function to print orbital elements
        self.plot_figure()                                                          # plot the figure of orbit

    def gibbs(self, data1, data2, data3):
        '''
        Gibb\'s Method

        Args:
            self : class variables
            data1 : position vector 1
            data2 : position vector 2
            data3 : position vector 3

        Returns:
            NIL
        '''

        v1 = data1.split()
        v2 = data2.split()
        v3 = data3.split()

        v1 = [float(v1[1])/1000, float(v1[2])/1000, float(v1[3])/1000]              # converting m to km
        v2 = [float(v2[1])/1000, float(v2[2])/1000, float(v2[3])/1000]
        v3 = [float(v3[1])/1000, float(v3[2])/1000, float(v3[3])/1000]

        mag_v1 = sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)                               # magnitude of position vector
        mag_v2 = sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        mag_v3 = sqrt(v3[0]**2 + v3[1]**2 + v3[2]**2)

        c12 = [v1[1]*v2[2] - v2[1]*v1[2], (-1)*(v1[0]*v2[2] - v2[0]*v1[2]), v1[0]*v2[1] - v2[0]*v1[1]]
        c23 = [v2[1]*v3[2] - v3[1]*v2[2], (-1)*(v2[0]*v3[2] - v3[0]*v2[2]), v2[0]*v3[1] - v3[0]*v2[1]]
        c31 = [v3[1]*v1[2] - v1[1]*v3[2], (-1)*(v3[0]*v1[2] - v1[0]*v3[2]), v3[0]*v1[1] - v1[0]*v3[1]]

        mag_c23 = sqrt(c23[0]**2 + c23[1]**2 + c23[2]**2)
        c23_unit = [float(c23[0]/mag_c23), float(c23[1]/mag_c23), float(c23[2]/mag_c23)]
        u_unit = [float(v1[0]/mag_v1), float(v1[1]/mag_v1), float(v1[2]/mag_v1)]
        coplanar = float(c23_unit[0]*u_unit[0]) + float(c23_unit[1]*u_unit[1]) + float(c23_unit[2]*u_unit[2])

        N = [float(mag_v1*c23[0] + mag_v2*c31[0] + mag_v3*c12[0]),
             float(mag_v1*c23[1] + mag_v2*c31[1] + mag_v3*c12[1]),
             float(mag_v1*c23[2] + mag_v2*c31[2] + mag_v3*c12[2])]
        mag_N = sqrt(N[0]**2 + N[1]**2 + N[2]**2)

        D = [c12[0] + c23[0] + c31[0], c12[1] + c23[1] + c31[1], c12[2] + c23[2] + c31[2]]
        mag_D = sqrt(D[0]**2 + D[1]**2 + D[2]**2)

        S = [(mag_v2-mag_v3)*v1[0] + (mag_v3-mag_v1)*v2[0] + (mag_v1-mag_v2)*v3[0],
             (mag_v2-mag_v3)*v1[1] + (mag_v3-mag_v1)*v2[1] + (mag_v1-mag_v2)*v3[1],
             (mag_v2-mag_v3)*v1[2] + (mag_v3-mag_v1)*v2[2] + (mag_v1-mag_v2)*v3[2]]

        Dv2 = [D[1]*v2[2] - v2[1]*D[2], (-1)*(D[0]*v2[2] - v2[0]*D[2]), D[0]*v2[1] - v2[0]*D[1]]
        div_Dv2 = Dv2/mag_v2
        val = sqrt(Gibbs.meu/(mag_N*mag_D))
        Dv2S = [div_Dv2[0] + S[0], div_Dv2[1] + S[1], div_Dv2[2] + S[2]]
        v = [val*Dv2S[0], val*Dv2S[1], val*Dv2S[2]]

        self.orbital_elements(v2, v)

    def orbital_elements(self, r, v):
        '''
        Finding orbital elements from the position and velocity vectors

        Args:
            self : class variables
            r : position vector
            v : velocity vector

        Returns:
            NIL
        '''

        mag_r = sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        mag_v = sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        vr = (v[0]*r[0] + v[1]*r[1] + v[2]*r[2])/mag_r

        h = [r[1]*v[2] - v[1]*r[2], (-1)*(r[0]*v[2] - v[0]*r[2]), r[0]*v[1] - v[0]*r[1]]
        mag_h = sqrt(h[0]**2 + h[1]**2 + h[2]**2)

        inclination = np.arccos(h[2]/mag_h)*(180/pi)

        N = [-h[1], -h[0], 0]
        mag_N = sqrt(N[0]**2 + N[1]**2 + N[2]**2)

        ascension = np.arccos(N[0]/mag_N)*(180/pi)
        if(N[1] < 0):
            ascension = 360 - ascension

        var1 = mag_v**2 - (Gibbs.meu/mag_r)
        var2 = r[0]*v[0] + r[1]*v[1] + r[2]*v[2]
        val1 = [var1*r[0], var1*r[1], var1*r[2]]
        val2 = [var2*v[0], var2*v[1], var2*v[2]]
        e = [val1[0] - val2[0], val1[1] - val2[1], val1[2] - val2[2]]
        e = [e[0]/Gibbs.meu, e[1]/Gibbs.meu, e[2]/Gibbs.meu]
        eccentricity = sqrt(e[0]**2 + e[1]**2 + e[2]**2)

        Ne = N[0]*e[0] + N[1]*e[1] + N[2]*e[2]
        perigee = np.arccos(Ne/(mag_N*eccentricity))*(180/pi)
        if(e[2] < 0):
            perigee = 360 - perigee

        er = e[0]*r[0] + e[1]*r[1] + e[2]*r[2]
        true_anomaly = np.arccos(er/(eccentricity*mag_r))*(180/pi)
        if(vr < 0):
            true_anomaly = 360 - true_anomaly

        r_p = float(mag_h**2/Gibbs.meu)*float(1/(1+eccentricity))
        r_a = float(mag_h**2/Gibbs.meu)*float(1/(1-eccentricity))
        semi_major_axis = (r_p + r_a)/2

        self.axis += semi_major_axis
        self.inc += inclination
        self.asc += ascension
        self.ecc += eccentricity
        self.per += perigee
        self.anom += true_anomaly

    def average(self):
        '''
        Average orbital elements to reduce noise

        Args:
            self : class variables

        Returns:
            NIL
        '''

        self.axis /= self.file_len
        self.inc /= self.file_len
        self.asc /= self.file_len
        self.ecc /= self.file_len
        self.per /= self.file_len
        self.anom /= self.file_len

    def display(self):
        '''
        Displaying calculated orbital elements

        Args:
            self : class variables

        Returns:
            NIL
        '''

        print "=== Orbital Elements ===\n"
        print "Semi-major Axis                                  : ", self.axis
        print "Inclination (degrees)                            : ", self.inc
        print "Right ascension of the ascending node (degrees)  : ", self.asc
        print "Eccentricity                                     : ", self.ecc
        print "Argument of perigee (degrees)                    : ", self.per
        print "True Anomaly                                     : ", self.anom

    def plot_figure(self):
        '''
        Plotting the orbital elements

        Args:
            self : class variables

        Returns:
            NIL
        '''

        graphics.plotOrbit(self.axis, self.ecc, self.inc, self.asc, self.per, self.anom)
        graphics.plotEarth()
        graphics.doDraw()

'''
gibbs_data1 is the dataset contains 4 columns, as
Time , x , y , z
'''
link = "https://raw.githubusercontent.com/aakash525/Orbital-Determinator/master/gibbs_data1.csv"
data = urllib2.urlopen(link)
d = data.read()
d = d.splitlines()

obj = Gibbs()
obj.read_file(d)
