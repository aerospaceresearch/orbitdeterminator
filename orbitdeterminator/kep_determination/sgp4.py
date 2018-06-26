"""
The code takes a TLE and computes state vectors for 8 hrs at every second
"""

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import math
import csv
import time
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

pi = np.pi
meu = 398600.4418
two_pi = 2*pi;
min_per_day = 1440

"""
class for SGP4 implementation
"""
class SGP4(object):

    @classmethod
    def find_year(self, year):
        """
        Returns year of launch of the satellite.

        Values in the range 00-56 are assumed to correspond to years in the range 2000 to 2056 while
        values in the range 57-99 are assumed to correspond to years in the range 1957 to 1999.

        Args:
            self : class variables
            year : last 2 digits of the year

        Returns:
            whole year number
        """

        if(year >=0 and year <=56):
            return year + 2000;
        else:
            return year + 1900;

    @classmethod
    def find_date(self, date):
        """
        Finds date of the year from the input (in number of days)

        Args:
            self : class variables
            date : Number of days

        Returns:
            date in format DD/MM/YYYY
        """

        year = int(self.find_year(int(''.join(date[0:2]))))
        day = int(''.join(date[2:5]))

        daysInMonth = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

        # If the year is Leap year or not
        if(year % 4 == 0):
            daysInMonth[1] = 29

        i = 0
        while(day > daysInMonth[i]):
            day -= daysInMonth[i]
            i += 1

        day += 1
        month = i+1
        print(str(year) + "/" + str(month) + "/" + str(day))

        return year, month, day

    @classmethod
    def find_time(self, time):
        """
        Finds date of the year from the input (in milliseconds)

        Args:
            self : class variables
            time : Time in milliseconds

        Returns:
            time in format HH:MM:SS
        """

        second = timedelta(float(time)/1000)
        time = datetime(1,1,1) + second

        hour = int(time.hour)
        minute = int(time.minute)
        second = int(time.second)
        print(str(hour) + ":" + str(minute) + ":" + str(second))

        return hour, minute, second

    @classmethod
    def julian_day(self, year, mon, day, hr, mts, sec):
        """
        Converts given timestamp into Julian form

        Args:
            self : class variables
            year : year number
            mon : month in year
            day : date in the month
            hr : hour
            mts : minutes in the hour
            sec : seconds in minute

        Returns:
            time in Julian form
        """
        return (367.0*year-7.0*(year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 +
          275.0 * mon // 9.0 + day + 1721013.5 +
          ((sec / 60.0 + mts) / 60.0 + hr) / 24.0)

    @classmethod
    def assure_path_exists(self, loc):
        """
        Creates a folder for output files if it does not exists

        Args:
            self : class variables
            loc : path to the folder

        Returns:
            NIL
        """
        # dir = os.path.dirname(path)
        if(os.path.exists(loc) == False):
            os.makedirs(loc)

    def maintain_data(self, line0, line1, line2):
        """
        Reads data, call propagation model and generates output files

        Args:
            self : class variables
            line0 : satellite name
            line1 : line 1 in TLE
            line2 : line 2 in TLE

        Returns:
            NIL
        """
        year, month, day = self.find_date(''.join(line1[18:23]))
        hour, minute, second = self.find_time(''.join(line1[24:32]))
        self.jd = self.julian_day(year, month, day, hour, minute, second)
        print(self.jd)

        self.xmo = float(''.join(line2[43:51])) * (pi/180)
        self.xnodeo = float(''.join(line2[17:25])) * (pi/180)
        self.omegao = float(''.join(line2[34:42])) * (pi/180)
        self.xincl = float(''.join(line2[8:16])) * (pi/180)
        self.eo = float('0.'+str(''.join(line2[26:33])))
        self.xno = float(''.join(line2[52:63]))*two_pi/min_per_day
        self.bstar = int(''.join(line1[53:59]))*(1e-5)*(10**int(''.join(line1[59:61])))

        ts = time.localtime(time.time())
        yr = ts.tm_year
        mth = ts.tm_mon
        day = ts.tm_mday
        hr = ts.tm_hour
        mts = ts.tm_min
        sec = ts.tm_sec

        suffix_date = str(yr) + "-" + str(mth) + "-" + str(day)
        suffix_time = str(hr) + ":" + str(mts) + ":" + str(sec)
        filename = line0 + "_" + suffix_date + "_" + suffix_time

        self.assure_path_exists("../output/")
        path = "../output/" + filename + ".csv"
        with open(path,'a') as myfile:
            writer = csv.writer(myfile)
            i = 0
            while(i < 28800):               # 28800
                # print(i)
                j = self.julian_day(yr, mth, day, hr, mts, sec)
                tsince = (j - self.jd)*min_per_day
                # print(i, j, tsince)
                pos, vel = self.propagation_model(tsince)
                pos = [1.54287467e+03,-2.54573872e+03,6.43448255e+03]
                # self.orbital_elements(pos, vel)
                # self.print_elements()
                # print(i, pos, vel)
                # print(vel)
                # timestamp = suffix_date + "-" + suffix_time
                data = [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]
                print(i, data)
                writer.writerows([data])
                yr, mth, day, hr, mts, sec = self.update_epoch(yr, mth, day, hr, mts, sec)
                i = i + 1

    @classmethod
    def update_epoch(self, yr, mth, day, hr, mts, sec):
        """
        Adds one second to the given time

        Args:
            self : class variables
            yr : year
            mth : month
            day : date
            hr : hour
            mts : minutes
            sec : seconds

        Returns:
            yr : year
            mth : month
            day : date
            hr : hour
            mts : minutes
            sec : seconds
        """
        sec += 1

        if(sec >= 60):
            sec = 0
            mts += 1

        if(mts >= 60):
            mts = 0
            hr += 1

        if(hr >= 24):
            hr = 0
            day += 1

        daysInMonth = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        if(yr % 4 == 0):
            daysInMonth[1] = 29

        if(day > daysInMonth[mth-1]):
            day = 1
            mth += 1

        if(mth > 12):
            mth = 1
            yr += 1

        return yr, mth, day, hr, mts, sec

    def propagation_model(self, tsince):
        """
        Computes state vectors at given time epoch

        Args:
            self : class variables
            tsince : time epoch

        Returns:
            pos : position vector
            vel : velocity vector
        """
        ae = 1
        tothrd = 2.0/3.0
        XJ3 = -2.53881e-6
        e6a = 1.0E-6
        xkmper = 6378.135
        ge = 398600.8           # Earth gravitational constant
        CK2 = 1.0826158e-3/2.0
        CK4 = -3.0*-1.65597e-6/8.0

        # Constants
        s = ae + 78 / xkmper
        qo = ae + 120 / xkmper
        xke = math.sqrt((3600 * ge)/(xkmper**3))
        qoms2t = ((qo-s)**2)**2
        temp2 = xke/self.xno
        a1 = temp2**tothrd
        cosio = math.cos(self.xincl)
        theta2 = cosio**2
        x3thm1 = 3*theta2-1
        eosq = self.eo**2
        betao2 = 1-eosq
        betao = math.sqrt(betao2)
        del1 = (1.5*CK2*x3thm1)/((a1**2)*betao*betao2)
        ao = a1*(1-del1*((1.0/3.0)+del1*(1+(134.0/81.0)*del1)))
        delo = 1.5*CK2*x3thm1/((ao**2)*betao*betao2)
        xnodp = (self.xno)/(1+delo)
        aodp = ao/(1-delo)

        # Initialization
        isimp = 0
        if((aodp*(1-self.eo)/ae) < (220.0/xkmper+ae)):
            isimp = 1

        s4 = s
        qoms24 = qoms2t
        perigee = (aodp*(1-self.eo)-ae)*xkmper
        if(perigee < 156):
            s4 = perigee - 78
            if(perigee <= 98):
                s4 = 20
            qoms24 = ((120-s4)*ae/xkmper)**4
            s4 = s4/xkmper+ae

        pinvsq = 1/((aodp**2)*(betao2**2))
        tsi = 1/(aodp-s4)
        eta = aodp*(self.eo)*tsi
        etasq = eta**2
        eeta = (self.eo)*eta
        psisq = abs(1-etasq)
        coef = qoms24*(tsi**4)
        coef1 = coef/(psisq**3.5)
        c2 = coef1*xnodp*(aodp*(1+1.5*etasq+eeta*(4+etasq))+0.75*CK2*tsi/psisq*x3thm1*(8+3*etasq*(8+etasq)))
        c1 = self.bstar*c2
        sinio = math.sin(self.xincl)
        a3ovk2 = -XJ3/CK2*(ae**3)
        c3 = coef*tsi*a3ovk2*xnodp*ae*sinio/self.eo
        x1mth2 = 1-theta2
        c4 = 2*xnodp*coef1*aodp*betao2*(eta*(2.0+0.5*etasq)+(self.eo)*(0.5+2*etasq)-2*CK2*tsi/(aodp*psisq)*(-3*x3thm1*(1-2*eeta+etasq*(1.5-0.5*eeta))+0.75*x1mth2*(2*etasq-eeta*(1+etasq))*math.cos(2*self.omegao)))
        c5 = 2*coef1*aodp*betao2*(1+2.75*(etasq+eeta)+eeta*etasq)
        theta4 = theta2**2
        temp1 = 3*CK2*pinvsq*xnodp
        temp2 = temp1*CK2*pinvsq
        temp3 = 1.25*CK4*(pinvsq**2)*xnodp
        xmdot = xnodp+0.5*temp1*betao*x3thm1+0.0625*temp2*betao*(13-78*theta2+137*theta4)
        x1m5th = 1-5*theta2
        omgdot = -0.5*temp1*x1m5th+0.0625*temp2*(7-114*theta2+395*theta4)+temp3*(3-36*theta2+49*theta4)
        xhdot1 = -temp1*cosio
        xnodot = xhdot1+(0.5*temp2*(4-19*theta2)+2*temp3*(3-7*theta2))*cosio
        omgcof = self.bstar*c3*math.cos(self.omegao)
        xmcof = -(2/3)*coef*(self.bstar)*ae/eeta
        xnodcf = 3.5*betao2*xhdot1*c1
        t2cof = 1.5*c1
        xlcof = 0.125*a3ovk2*sinio*(3+5*cosio)/(1+cosio)
        aycof = 0.25*a3ovk2*sinio
        delmo = (1+eta*math.cos(self.xmo))**3
        sinmo = math.sin(self.xmo)
        x7thm1 = 7*theta2-1
        if(isimp == 0):
            c1sq = c1**2
            d2 = 4*aodp*tsi*c1sq
            temp = d2*tsi*c1/3
            d3 = (17*aodp+s4)*temp
            d4 = 0.5*temp*aodp*tsi*(221*aodp+31*4)*c1
            t3cof = d2+2*c1sq
            t4cof = 0.25*(3*d3+c1*(12*d2+10*c1sq))
            t5cof = 0.2*(3*d4+12*c1*d3+6*(d2**2)+15*c1sq*(2*d2+c1sq))

        xmdf = self.xmo+xmdot*tsince
        omgadf = self.omegao+omgdot*tsince
        xnoddf = self.xnodeo+xnodot*tsince
        omega = omgadf
        xmp = xmdf
        tsq = tsince**2
        xnode = xnoddf+xnodcf*tsq
        tempa = 1 - c1*tsince
        tempe = self.bstar*c4*tsince
        templ = t2cof*tsq
        if(isimp == 0):
            delomg = omgcof*tsince
            delm = xmcof*(((1+eta*math.cos(xmdf))**3)-delmo)
            temp = delomg+delm
            xmp = xmdf+temp
            omega = omgadf-temp
            tcube = tsq*tsince
            tfour = tsince*tcube
            tempa = tempa-d2*tsq-d3*tcube-d4*tfour
            tempe = tempe+self.bstar*c5*(math.sin(xmp)-sinmo)
            templ = templ+t3cof*tcube+tfour*(t4cof+tsince*t5cof)

        a = aodp*(tempa**2)
        e = self.eo-tempe
        xl = xmp+omega+xnode+xnodp*templ
        beta = math.sqrt(1-e**2)
        xn = xke/(a**1.5)

        axn = e*math.cos(omega)
        temp = 1/(a*(beta**2))
        xll = temp*xlcof*axn
        aynl = temp*aycof
        xlt = xl+xll
        ayn = e*math.sin(omega)+aynl
        diff = xlt - xnode
        capu = diff - math.floor(diff/two_pi) * two_pi
        if(capu < 0):
            capu = capu + two_pi
        temp2 = capu

        i = 1
        while(1):
            sinepw = math.sin(temp2)
            cosepw = math.cos(temp2)
            temp3 = axn*sinepw
            temp4 = ayn*cosepw
            temp5 = axn*cosepw
            temp6 = ayn*sinepw
            epw = (capu-temp4+temp3-temp2)/(1-temp5-temp6)+temp2
            temp7 = temp2
            temp2 = epw
            i = i + 1
            if((i>10) | (abs(epw-temp7)<=e6a)):
                break

        ecose = temp5+temp6
        esine = temp3-temp4
        elsq = axn**2 + ayn**2
        temp = 1-elsq
        pl = a*temp
        r = a*(1-ecose)
        temp1 = 1/r
        rdot = xke*math.sqrt(a)*esine*temp1
        rfdot = xke*math.sqrt(pl)*temp1
        temp2 = a*temp1
        betal = math.sqrt(temp)
        temp3 = 1/(1+betal)
        cosu = temp2*(cosepw-axn+ayn*esine*temp3)
        sinu = temp2*(sinepw-ayn-axn*esine*temp3)
        u = math.atan2(sinu, cosu)
        if(u < 0):
            u = u + two_pi

        sin2u = 2*sinu*cosu
        cos2u = 2*(cosu**2)-1
        temp = 1/pl
        temp1 = CK2*temp
        temp2 = temp1*temp

        rk = r*(1-1.5*temp2*betal*x3thm1)+0.5*temp1*x1mth2*cos2u
        uk = u-0.25*temp2*x7thm1*sin2u
        xnodek = xnode+1.5*temp2*cosio*sin2u
        xinck = self.xincl+1.5*temp2*cosio*sinio*cos2u
        rdotk = rdot-xn*temp1*x1mth2*sin2u
        rfdotk = rfdot+xn*temp1*(x1mth2*cos2u+1.5*x3thm1)

        MV = [-math.sin(xnodek)*math.cos(xinck), math.cos(xnodek)*math.cos(xinck), math.sin(xinck)]
        NV = [math.cos(xnodek), math.sin(xnodek), 0]

        UV = [0, 0, 0]
        VV = [0, 0, 0]
        for i in range(3):
            UV[i] = MV[i]*math.sin(uk) + NV[i]*math.cos(uk)
            VV[i] = MV[i]*math.cos(uk) - NV[i]*math.sin(uk)

        pos = [0, 0, 0]
        vel = [0, 0, 0]
        for i in range(3):
            pos[i] = rk*UV[i]*xkmper
            vel[i] = (rdotk*UV[i] + rfdotk*VV[i])*xkmper/60

        return pos, vel

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
    def vec_multiply(self, a, b):
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
    def matrix_multiply(self, a, b):
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

    def orbital_elements(self, pos, vel):
        """
        Finding orbital elements from the position and velocity vectors

        Args:
            self : class variables
            r : position vector
            v : velocity vector

        Returns:
            NIL
        """

        mag_pos = self.magnitude(pos)
        mag_vel = self.magnitude(vel)
        radial_vel = self.vec_multiply(pos, vel)/mag_pos
        ang_momentum = self.matrix_multiply(pos, vel)
        mag_ang_momentum = self.magnitude(ang_momentum)
        inclination = np.arccos(ang_momentum[2]/mag_ang_momentum)*(180/pi)

        N = self.matrix_multiply([0,0,1], ang_momentum)
        mag_N = self.magnitude(N)
        ascension = np.arccos(N[0]/mag_N)*(180/pi)
        if(N[1] < 0):
            ascension = 360 - ascension

        var1 = mag_vel**2 - meu/mag_pos
        var2 = [pos[0]*var1, pos[1]*var1, pos[2]*var1]
        var3 = mag_pos*radial_vel
        var4 = [vel[0]*var3, vel[1]*var3, vel[2]*var3]
        e = [(var2[0]-var4[0])/meu, (var2[1]-var4[1])/meu, (var2[2]-var4[2])/meu]
        eccentricity = self.magnitude(e)

        orbital_perigee = np.arccos(self.vec_multiply(N,e)/(mag_N*eccentricity))*(180/pi)
        if(e[2] < 0):
            orbital_perigee = 360 - orbital_perigee

        true_anomaly = np.arccos(self.vec_multiply(e,pos)/(eccentricity*mag_pos))*(180/pi)
        if(radial_vel < 0):
            true_anomaly = 360 - true_anomaly

        r_p = float(mag_ang_momentum**2/(meu*(1+eccentricity)))
        r_a = float(mag_ang_momentum**2/(meu*(1-eccentricity)))
        semi_major_axis = (r_p+r_a)/2

        self.axis = semi_major_axis
        self.inc = inclination
        self.asc = ascension
        self.ecc = eccentricity
        self.per = orbital_perigee
        self.anom = true_anomaly
        self.print_elements()

    # def display(self):
    #     print(self.xmo)
    #     print(self.xnodeo)
    #     print(self.omegao)
    #     print(self.xincl)
    #     print(self.eo)
    #     print(self.xno)
    #     print(self.bstar)

    def print_elements(self):
        print ("\n=== Orbital Elements ===")
        print ("Semi-major Axis                                  : ", self.axis)
        print ("Inclination (degrees)                            : ", self.inc)
        print ("Right ascension of the ascending node (degrees)  : ", self.asc)
        print ("Eccentricity                                     : ", self.ecc)
        print ("Argument of perigee (degrees)                    : ", self.per)
        print ("True Anomaly                                     : ", self.anom)

if __name__ == "__main__":
    page = requests.get("https://www.celestrak.com/NORAD/elements/cubesat.txt")
    soup = BeautifulSoup(page.content, 'html.parser')
    tle = list(soup.children)
    tle = tle[0].splitlines()

    # count = len(tle)
    # for i in range(0,count,3):
    #     print(str(i/3) + " - " + tle[i])
    #     obj = SGP4()
    #     obj.maintain_data(tle[i].replace(" ", ""), tle[i+1], tle[i+2])
    #     del(obj)

    line1 = tle[1]
    line2 = tle[2]
    line1 = "1 35933U 09051C   18170.11271880  .00000090  00000-0  31319-4 0  9993"
    line2 = "2 35933  98.5496 322.8685 0005266 206.2829 153.8102 14.56270197463823"

    obj = SGP4()
    obj.maintain_data(tle[0].replace(" ", ""), line1, line2)
    # pos = [-1.57548492e+03, 3.58011715e+03, 5.91547730e+03]
    # vel = [2.95658397e+00, -5.52287181e+00, 4.12343017e+00]
    # obj.orbital_elements(pos,vel)
    print(line1)
    print(line2)
    del(obj)

    # [1542.87467, -2545.73872, 6434.48255]
    # [3.328721982844391, 0.98770565136204946, 6.5935970441656533]

    # [1542.87467, -2545.73872, 6434.48255]
    # [4.4692636257169811, -0.88389096855806137, -5.8817723039953069]
