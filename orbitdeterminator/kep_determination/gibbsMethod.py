'''
Converts a set of three psoition vectors into state vector using Gibb's method
from which orbital elements can be found easily.
'''

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
        '''
        Type casts the input data for the ease of use.

        Converts the values of the input list with string datatype into float
        for the ease of furthur computation.

        Args:
            vec (list): input vector

        Returns:
            list: vector converted to float values
        '''
        a = 0.0
        b = 0.0
        c = 0.0
        try:
            a = float(vec[1])
            b = float(vec[2])
            c = float(vec[3])
        except ValueError:
            print("Double check file format!")
        return([a, b, c])

    @classmethod
    def find_length(self, path):
        '''
        Finds the length of the input file.

        Calculates the length of the file with the given path containing all the
        position vectors. File should contain a header line describing all of its
        attributes in a single line only. This function removes the header line
        before it calculates file length. If the file does not contains the header
        line then the first line of data will get removed and you have to add 1
        to the output after it returns the result.

        Args:
            path (str): file path

        Returns:
            int: length of file
        '''
        myfile = open(path, 'r')

        #To remove all headers
        while(1):
            pointer = myfile.tell()
            tempstr = myfile.readline()
            # EOF reached
            if(tempstr == ""):
                break
            # If it contains only those literals required to make a number then this line might be start of data
            if(all(c.isdigit() or c == ',' or c == '.' or c == ' ' or c == '\t' or c == '-' or c == '+' or c == 'e' or c == 'E' or c == '\n' or c == '\r' for c in tempstr)):
                # Undo read and break so that this data gets included in size
                myfile.seek(pointer)
                break

        # Now read data lines and find size
        size = 0
        while(myfile.readline()):
            size = size + 1

        return size

    def read_file(self, path):
        '''
        Invokes the Gibb's implementation and stores the result in a list.

        Read the file with the given path and forms a set of three position vectors
        then applies Gibb's Method on that set and computes state vector for every
        set. After computing state vectors it stores the result into a list. Now,
        these state vectors can be used to find orbital elements.

        Args:
            path (str): path to input file

        Returns:
            numpy.ndarray: list of all pair of position and velocity vector
        '''
        myfile = open(path, 'r')
        #To remove all headers
        while(1):
            pointer = myfile.tell()
            tempstr = myfile.readline()
            # EOF reached
            if(tempstr == ""):
                break
            # If it contains only those literals required to make a number then this line might be start of data
            if(all(c.isdigit() or c == ',' or c == '.' or c == ' ' or c == '\t' or c == '-' or c == '+' or c == 'e' or c == 'E' or c == '\n' or c == '\r' for c in tempstr)):
                # Undo read and break so that this data gets included in size
                myfile.seek(pointer)
                break

        kep = np.zeros((6, 1))
        size = self.find_length(path)
        upto = size-2                    # size-2
        # Size might not be enough
        try:
            final = np.zeros((upto, 6))
        except ValueError:
            print("Enough data samples not present")

        # Read line reads with '\n' as well which should not get split
        str1 = myfile.readline().replace('\n', '')
        str2 = myfile.readline().replace('\n', '')
        # Lines end with "\r\n" in windows, so for safety measure
        str1 = str1.replace('\r', '')
        str2 = str2.replace('\r', '')

        r1 = []
        r2 = []
        # Check if files are comma delimited
        if("," in str1):
            str1 = str1.replace(' ', '')
            str2 = str2.replace(' ', '')
            str1 = str1.replace('\t', '')
            str2 = str2.replace('\t', '')
            # Split on files delimited with comma
            r1 = self.convert_list(re.split(',', str1))
            r2 = self.convert_list(re.split(',', str2))
        else:
            # Split on files delimited with tabs and space
            r1 = self.convert_list(re.split('\t|\s', str1))
            r2 = self.convert_list(re.split('\t|\s', str2))

        i = 0
        while(i < upto):
            str3 = myfile.readline().replace('\n', '')
            str3 = str3.replace('\r', '')
            r3 = []
            # Check if files are comma delimited
            if("," in str1):
                str3 = str1.replace(' ', '')
                str3 = str1.replace('\t', '')
                # Split on files delimited with comma
                r3 = self.convert_list(re.split(',', str3))
            else:
                # Split on files delimited with tabs and space
                r3 = self.convert_list(re.split('\t|\s', str3))
            v2 = self.gibbs(r1, r2, r3)
            ele = self.orbital_elements(r2, v2)
            # Add to keplerian elements to later on find average
            for j in range(6):
                kep[j, 0] += ele[j]
            data = [r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]]
            final[i,:] = data

            r1 = r2
            r2 = r3
            i = i + 1

        # Now find average and return data        
        for j in range(6):
            kep[j, 0] /= upto
        return kep

        # Returning r and v array for now
        # return final

    @classmethod
    def magnitude(self, vec):
        '''
        Computes magnitude of the input vector.

        Args:
            vec (list): vector

        Returns:
            float: magnitude of vector
        '''
        mag_vec = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        return mag_vec

    @classmethod
    def dot_product(self, a, b):
        '''
        Computes dot product of two vectors. Multiplies corresponding axis with
        each other and then adds them. Returns a single value.

        Args:
            a (list/array): first vector
            b (list/array): second vector

        Returns:
            float: dot product of given vectors
        '''
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

    @classmethod
    def cross_product(self, a, b):
        '''
        Computes cross product of the given vectors. Returns a vector.

        Args:
            a (list/array): first vector
            b (list/array): second vector

        Returns:
            list/array: cross product of given vectors
        '''
        return [a[1]*b[2] - b[1]*a[2], (-1)*(a[0]*b[2] - b[0]*a[2]), a[0]*b[1] - b[0]*a[1]]

    @classmethod
    def operate_vector(self, a, b, flag):
        '''
        Adds or subtracts input vectors based on the flag value.

        If flag is 1 then add both vectors with corresponding values else if
        flag is 0 (zero) then subtract two vectors with corresponding values.
        Returns a vector.

        Args:
            a (list/array): first vector
            b (list/array): second vector
            flag (int): checks for operation (addition/subtraction)

        Returns:
            list/array: sum/difference of vector based on flag value
        '''
        if(flag == 1):
            return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
        elif(flag == 0):
            return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

    @classmethod
    def unit(self, vec):
        '''
        Finds unit vector of the given vector. Divides each value of vector by
        its magnitude. Returns a vector.

        Args:
            vec (list): input vector

        Returns:
            list: unit vector
        '''
        mag = self.magnitude(vec)
        return [i/mag for i in vec]

    @classmethod
    def gibbs(self, r1, r2, r3):
        '''
        Computes state vector from the given set of three position vectors using
        Gibb's implementation.

        Computes velocity vector (part of state vector) using Gibb's Method
        and takes r2 (input argument) as its position vector (part of state
        vector). Both combined forms state vector.

        Args:
            r1 (list): first position vector
            r2 (list): second position vector
            r3 (list): third position vector

        Returns:
            list: velocity vector
        '''
        mag_r1 = self.magnitude(r1)
        mag_r2 = self.magnitude(r2)
        mag_r3 = self.magnitude(r3)

        c12 = self.cross_product(r1, r2)
        c23 = self.cross_product(r2, r3)
        c31 = self.cross_product(r3, r1)

        # '''For checking colplanarity'''
        # unit_c23 = self.unit(c23)
        # unit_r1 = self.unit(r1)
        # check = self.dot_product(unit_r1, unit_c23)

        r1c23 = [mag_r1*i for i in c23]
        r2c31 = [mag_r2*i for i in c31]
        r3c12 = [mag_r3*i for i in c12]
        N = [r1c23[0]+r2c31[0]+r3c12[0], r1c23[1]+r2c31[1]+r3c12[1], r1c23[2]+r2c31[2]+r3c12[2]]
        mag_N = self.magnitude(N)

        D = self.operate_vector(c12, self.operate_vector(c23, c31, 1), 1)
        mag_D = self.magnitude(D)

        vec1 = [(mag_r2-mag_r3)*i for i in r1]
        vec2 = [(mag_r3-mag_r1)*i for i in r2]
        vec3 = [(mag_r1-mag_r2)*i for i in r3]
        S = self.operate_vector(vec1, self.operate_vector(vec2, vec3, 1), 1)

        term1 = math.sqrt(meu/(mag_N*mag_D))
        var1 = self.cross_product(D, r2)
        var2 = [i/mag_r2 for i in var1]
        term2 = self.operate_vector(var2, S, 1)
        v2 = [term1*i for i in term2]

        return v2

    @classmethod
    def orbital_elements(self, r, v):
        '''
        Computes orbital elements from state vector.

        Orbital elements is a set of six parameters which are semi-major axis,
        inclination, right ascension of the ascending node, eccentricity,
        argument of perigee and mean anomaly.

        Args:
            r (list): position vector
            v (list): velocity vector

        Returns:
            list: set of six orbital elements
        '''
        mag_r = self.magnitude(r)
        mag_v = self.magnitude(v)
        vr = self.dot_product(r, v)/mag_r
        h = self.cross_product(r, v)
        mag_h = self.magnitude(h)
        # Requires further research, not put in try block because one set should not affect entire calculation
        if((h[2]/mag_h) <= 1 and (h[2]/mag_h) >= -1):
            inclination = math.acos(h[2]/mag_h)*(180/pi)
        elif((h[2]/mag_h) > 1):
            inclination = 0
        else:
            inclination = 180

        N = self.cross_product([0,0,1], h)
        mag_N = self.magnitude(N)

        ascension = math.acos(N[0]/mag_N)*(180/pi)
        if(N[1] < 0):
            ascension = 360 - ascension

        var1 = [(mag_v**2 - meu/mag_r)*i for i in r]
        var2 = [self.dot_product(r, v)*i for i in v]
        vec = self.operate_vector(var1, var2, 0)
        eccentricity = [i/meu for i in vec]
        mag_e = self.magnitude(eccentricity)

        # Requires further research, not put in try block because one set should not affect entire calculation
        if((self.dot_product(N,eccentricity)/(mag_N*mag_e)) <= 1 and (self.dot_product(N,eccentricity)/(mag_N*mag_e)) >= -1):
            perigee = math.acos(self.dot_product(N,eccentricity)/(mag_N*mag_e))*(180/pi)
        elif((self.dot_product(N,eccentricity)/(mag_N*mag_e)) > 1):
            perigee = 0
        else:
            perigee = 180
        if(eccentricity[2] < 0):
            perigee = 360 - perigee

        # Requires further research, not put in try block because one set should not affect entire calculation
        if((self.dot_product(eccentricity,r)/(mag_e*mag_r)) <= 1 and (self.dot_product(eccentricity,r)/(mag_e*mag_r)) >= -1):
            anomaly = math.acos(self.dot_product(eccentricity,r)/(mag_e*mag_r))*(180/pi)
        elif(self.dot_product(eccentricity,r)/(mag_e*mag_r) > 1):
            anomaly = 0
        else:
            anomaly = 180
        if(vr < 0):
            anomaly = 360 - anomaly

        rp = mag_h**2/(meu*(1+mag_e))
        ra = mag_h**2/(meu*(1-mag_e))
        axis = (rp+ra)/2

        # Following format trend from test_gibbsMethod file
        # return [axis, inclination, ascension, mag_e, perigee, anomaly]
        # Following format trend from lamberts_kalman file
        return [axis, mag_e, inclination, perigee, ascension, anomaly]

def gibbs_get_kep(dataset):
    ''' 
    Determines keplerian elements using Gibbs 3 vector method.

    Args:
        data(nx3 numpy array): A numpy array of points in the format [x y z].

    Returns:
        (kep) - The keplerian elements as 1x6 numpy array.

        For the keplerian elements:
        kep[0] - semi-major axis (in whatever units the data was provided in)
        kep[1] - eccentricity
        kep[2] - inclination (in degrees)
        kep[3] - argument of periapsis (in degrees)
        kep[4] - right ascension of ascending node (in degrees)
        kep[5] - true anomaly of the first row in the data (in degrees)
    '''
    final = []
    kep = []
    r1 = [dataset[0,0], dataset[0,1], dataset[0,2]]
    r2 = [dataset[1,0], dataset[1,1], dataset[1,2]]

    i = 0
    upto = dataset.shape[0] - 2
    # Size might not be enough
    try:
        final = np.zeros((upto, 6))
        kep = np.zeros((upto, 6))
    except ValueError:
        print("Enough data samples not present")

    while(i < upto):
        r3 = [dataset[i+2,0], dataset[i+2,1], dataset[i+2,2]]
        obj1 = Gibbs()
        v2 = obj1.gibbs(r1, r2, r3)
        ele = obj1.orbital_elements(r2, v2)
        del(obj1)
        # Add to keplerian elements to later on find average
        kep[i, :] = ele
        data = [r2[0], r2[1], r2[2], v2[0], v2[1], v2[2]]
        final[i,:] = data

        r1 = r2
        r2 = r3
        i = i + 1
    np.savetxt("data_from_gibb.csv", kep, delimiter=", ")
    return kep

# if __name__ == "__main__":
#     filename = "ISS.csv"
#     path = "../example_data/" + filename

#     obj = Gibbs()
#     vector = obj.read_file(path)
#     print(vector)
#     del(obj)
