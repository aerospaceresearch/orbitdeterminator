import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import math
import re

sqrt = np.sqrt
pi = np.pi
meu = 398600.4418

class Gibbs(object):

    @classmethod
    def convert_list(self, vec):
        return [float(vec[1]), float(vec[2]), float(vec[3])]

    def read_file(self, path):
        myfile = open(path, 'r')
        myfile.readline()

        r1 = self.convert_list(re.split('\t|\n', myfile.readline()))
        r2 = self.convert_list(re.split('\t|\n', myfile.readline()))
        r3 = self.convert_list(re.split('\t|\n', myfile.readline()))

        r1 = [-294.32, 4265.1, 5986.7]
        r2 = [-1365.5, 3637.6, 6346.8]
        r3 = [-2940.3, 2473.7, 6555.8]
        # print("r1 ", r1)
        # print("r2 ", r2)
        # print("r3 ", r3)

        v2 = self.gibbs(r1, r2, r3)
        self.orbital_elements(r2, v2)
        # for i in range(7998):
        #     self.gibbs(r1, r2, r3)
        #     r1 = r2
        #     r2 = r3
        #     r3 = re.split('\t|\n', myfile.readline())

    @classmethod
    def magnitude(self, vec):
        """
        Computes magnitude of a given vector

        Args:
            self : class variables
            vec : vector

        Returns:
            magnitude of vector
        """
        mag_vec = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        return mag_vec

    @classmethod
    def dot_product(self, a, b):
        """
        Dot product of two vectors

        Args:
            self : class variables
            a : first vector
            b : second vector

        Returns:
            dot product
        """
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

    @classmethod
    def cross_product(self, a, b):
        """
        Cross product of two vectors

        Args:
            self : class variables
            a : first vector
            b : second vector

        Returns:
            cross product
        """
        return [a[1]*b[2] - b[1]*a[2], (-1)*(a[0]*b[2] - b[0]*a[2]), a[0]*b[1] - b[0]*a[1]]

    @classmethod
    def vector_sum(self, a, b, flag):
        if(flag == 1):
            return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
        elif(flag == 0):
            return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

    @classmethod
    def unit(self, vec):
        mag = self.magnitude(vec)
        return [i/mag for i in vec]

    def gibbs(self, r1, r2, r3):
        mag_r1 = self.magnitude(r1)
        mag_r2 = self.magnitude(r2)
        mag_r3 = self.magnitude(r3)
        # print("mag_r1", mag_r1)
        # print("mag_r2", mag_r2)
        # print("mag_r3", mag_r3)

        c12 = self.cross_product(r1, r2)
        c23 = self.cross_product(r2, r3)
        c31 = self.cross_product(r3, r1)
        # print("c12", c12)
        # print("c23", c23)
        # print("c31", c31)

        # """For checking colplanarity"""
        # unit_c23 = self.unit(c23)
        # unit_r1 = self.unit(r1)
        # check = self.dot_product(unit_r1, unit_c23)
        # print("check", check)

        r1c23 = [mag_r1*i for i in c23]
        r2c31 = [mag_r2*i for i in c31]
        r3c12 = [mag_r3*i for i in c12]
        N = [r1c23[0]+r2c31[0]+r3c12[0], r1c23[1]+r2c31[1]+r3c12[1], r1c23[2]+r2c31[2]+r3c12[2]]
        mag_N = self.magnitude(N)
        # print("N ", N)
        # print("mag_N", mag_N)

        D = self.vector_sum(c12, self.vector_sum(c23, c31, 1), 1)
        mag_D = self.magnitude(D)
        # print("D", D)
        # print("mag_D", mag_D)

        vec1 = [(mag_r2-mag_r3)*i for i in r1]
        vec2 = [(mag_r3-mag_r1)*i for i in r2]
        vec3 = [(mag_r1-mag_r2)*i for i in r3]
        S = self.vector_sum(vec1, self.vector_sum(vec2, vec3, 1), 1)
        # print("S", S)

        term1 = math.sqrt(meu/(mag_N*mag_D))
        var1 = self.cross_product(D, r2)
        var2 = [i/mag_r2 for i in var1]
        term2 = self.vector_sum(var2, S, 1)
        v2 = [term1*i for i in term2]
        # print("v2", v2)

        return v2

    def orbital_elements(self, r, v):
        r = [-6045, -3490, 2500]
        v = [-3.457, 6.618, 2.533]
        # print("r", r)
        # print("v", v)

        mag_r = self.magnitude(r)
        mag_v = self.magnitude(v)
        vr = self.dot_product(r, v)/mag_r
        h = self.cross_product(r, v)
        mag_h = self.magnitude(h)
        inclination = math.acos(h[2]/mag_h)*(180/pi)
        # print("inclination", inclination)

        N = self.cross_product([0,0,1], h)
        mag_N = self.magnitude(N)

        ascension = math.acos(N[0]/mag_N)*(180/pi)
        if(N[1] < 0):
            ascension = 360 - ascension
        # print("ascension", ascension)

        var1 = [(mag_v**2 - meu/mag_r)*i for i in r]
        var2 = [self.dot_product(r, v)*i for i in v]
        vec = self.vector_sum(var1, var2, 0)
        eccentricity = [i/meu for i in vec]
        mag_e = self.magnitude(eccentricity)
        # print("eccentricity", eccentricity)

        perigee = math.acos(self.dot_product(N,eccentricity)/(mag_N*mag_e))*(180/pi)
        if(eccentricity[2] < 0):
            perigee = 360 - perigee
        # print("perigee", perigee)

        anomaly = math.acos(self.dot_product(eccentricity,r)/(mag_e*mag_r))*(180/pi)
        if(vr < 0):
            anomaly = 360 - anomaly
        # print("anomaly", anomaly)

        rp = mag_h**2/(meu*(1+mag_e))
        ra = mag_h**2/(meu*(1-mag_e))
        axis = (rp+ra)/2
        # print("axis", axis)

if __name__ == "__main__":
    filename = "orbit_simulated_a6801000.0_ecc0.000515_inc134.89461080388952_raan112.5156_aop135.0415_ta225.1155_jit0.0_dt1.0_gapno_1502628669.3819425.csv"
    path = "../example_data/" + filename

    obj = Gibbs()
    obj.read_file(path)
