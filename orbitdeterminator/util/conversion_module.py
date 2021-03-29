import numpy as np

from math import pi, sin, cos, radians, degrees, floor, sqrt, atan2
from datetime import datetime, timedelta
from sys import stdin

from math import *
from pyorbital import dt2np
from astropy import constants as cts
from astropy import units as u
from astropy.constants import G, M_earth, R_earth



mu = G.value*M_earth.value
Re = R_earth.value
rad_earth= 6378.13  # Radius
r_test = np.array([Re + 600.0*1000, 0, 50])
v_test = np.array([0, 6.5 * 1000, 0])
t = 0
M_fact= 7.292115E-5
mu_Sun = cts.GM_sun.to(u.Unit('au3 / day2')).value
F= 1 / 298.257223563  # WGS-84
mu_Earth = cts.GM_earth.to(u.Unit('km3 / s2')).value



def julian_day_from_utc(utc_time):
    """Returns julian day.
    """
    return days_since_2000(utc_time) + 2451545



def cartesian_to_spherical(data):
    '''
    Takes as an input a data set containing points in cartesian format (time, x, y, z) and returns the computed
    spherical coordinates (time, azimuth, elevation, r)

    Args:
        data (np array): containing the cartesian coordinates in format of (time, x, y, z)

    Returns:
        np array: spherical coordinates in format of (time, azimuth, elevation, r)
    '''

    for i in range(0, len(data)):
        z = data[i, 3]
        x = data[i, 1]
        y = data[i, 2]
        

        result = data

        elevation = atan2(z, sqrt(x**2 + y**2))
        azimuth = atan2(y, x)

        r = sqrt(x**2 + y**2 + z**2)

        result[i, 2] = elevation
        result[i, 3] = r        
        result[i, 1] = azimuth


    return result 



def checksum(line):
    """Compute the TLE checksum."""
    return (sum((int(c) if c.isdigit() else c == '-') for c in line[0:-1]) % 10) == int(line[-1])



def getTrueAnomaly(ecc, EA):
	fak = sqrt(1.0 - ecc * ecc)
	return degrees(atan2(fak * sin(EA), cos(EA) - ecc))



def days_from_dt(dt):
    """Returns dar from dt.
    """
    return dt / np.timedelta64(1, 'D')



def getEccentricAnomaly(ecc, MA):
    """Returns eccentric anomaly from eccentricity and mean anomaly.
    """    
    precision = 1e-6
    iterLimit = 50

    i = 0

    EA = MA

    F = EA - ecc * sin(MA) - MA

    while ((abs(F) > precision) and (i < iterLimit)):

	    EA = EA - F / (1.0 - ecc * cos(EA))
	    F = EA - ecc * sin(EA) - MA

	    i += 1

    return degrees(EA)



def days_since_2000(utc_time):
    """Returns the days since year 2000.
    """
    return days_from_dt(dt2np(utc_time) - np.datetime64('2000-01-01T12:00')) 



def MM2SMA(MM):
	MU = 398600.4418
	# Convert Mean motion from revs/day to rad/s
	MM = MM*((2*pi)/86400)
	return (MU/(MM**2))**(1.0/3.0)



def print_norad_elements():
    """Prints elements from tle.
    """
    print ("Enter TLE including object title:")

    TLE=[]
    TLE.append(stdin.readline().strip())
    TLE.append(stdin.readline().strip())
    TLE.append(stdin.readline().strip())

    if TLE[1][:2] != '1 ' or checksum(TLE[1]) == False:
	    print ("Not a TLE")
	    exit()
    if TLE[2][:2] != '2 'or checksum(TLE[2]) == False:
	    print ("Not a TLE")
	    exit()

    SatName = TLE[0]
    (line,SAT,Desgnator,TLEEpoch,MM1,MM2,BSTAR,EType,ElementNum) = TLE[1].split()
    (line,SATNum,Inc,RAAN,Ecc,AoP,MA,MM) = TLE[2].split()[:8]
    EpochY = int(TLEEpoch[:2])
    if EpochY > 56:
	    EpochY+=1900
    else:
	    EpochY+=2000
    EpochD = float(TLEEpoch[2:])
    MA = float(MA)
    MM = float(MM)
    Ecc = float('0.'+Ecc)
    SMA = MM2SMA(MM)
    EA = getEccentricAnomaly(Ecc, radians(MA))
    TA = getTrueAnomaly(Ecc, radians(EA))
    Epoch = (datetime(EpochY-1,12,31) + timedelta(EpochD)).strftime("%d %b %Y %H:%M:%S.%f")[:-3]

    print("Year:",EpochY,"\nDay:",EpochD,"\nInclination:",Inc,"\nRAAN:",RAAN,"\nEccentricity:",Ecc)
    print("AoP:",AoP,"\nMean Anomaly:",MA,"\nEcc. Anomaly:", EA,"\nTrue Anomaly:", TA, "\nMM:",MM, "\nSemi Magor Axis:", SMA)
    print("Epoch:", Epoch)

    print("\nCreate Spacecraft "+SatName+";\n" +
	    	"GMAT "+SatName+".Id = '"+SATNum+"';\n" +
		    "GMAT "+SatName+".DateFormat = UTCGregorian;\n" +
    		"GMAT "+SatName+".Epoch = '"+Epoch+"';\n" +
    		"GMAT "+SatName+".CoordinateSystem = EarthMJ2000Eq;\n" +
	    	"GMAT "+SatName+".DisplayStateType = Keplerian;\n" +
    		"GMAT "+SatName+".SMA = "+str(SMA)+";\n" +
    		"GMAT "+SatName+".ECC = "+str(Ecc)+";\n" +
    		"GMAT "+SatName+".INC = "+str(Inc)+";\n" +
    		"GMAT "+SatName+".RAAN = " + str(RAAN) + ";\n" +
    		"GMAT "+SatName+".AOP = " + str(AoP) + ";\n" +
    		"GMAT "+SatName+".TA = "+str(TA)+";\n")



def g_m_sidereal_time(utc_time):
    """Greenwich mean sidereal utc_time, in radians.
    """
    ut1 = days_since_2000(utc_time) / 36525.0
    theta = 67310.54841 + ut1 * (876600 * 3600 + 8640184.812866 + ut1 * (0.093104 - ut1 * 6.2 * 10e-6))
    return np.deg2rad(theta / 240.0) % (2 * np.pi)



def local_hour_angle(utc_time, longitude, right_ascension):
    """Returns hour angle at utc_time for the given longitude and right_ascension longitude.
    """
    return local_mean_sidereal_time(utc_time, longitude) - right_ascension



def sun_ra_dec(utc_time):
    """Returns right ascension and declination of the sun at utc_time.
    """
    jdate = days_since_2000(utc_time) / 36525.
    
    eps = np.deg2rad(23.0 + 26.0 / 60.0 + 21.448 / 3600.0 -(46.8150 * jdate + 0.00059 * jdate * jdate - 0.001813 * jdate * jdate * jdate) / 3600)
    
    eclon = sun_ecliptic_long(utc_time)

    x__ = np.cos(eclon)
    y__ = np.cos(eps) * np.sin(eclon)
    z__ = np.sin(eps) * np.sin(eclon)
    r__ = np.sqrt(1.0 - z__ * z__)
    
    dec = np.arctan2(z__, r__)
    
    right_ascension = 2 * np.arctan2(y__, (x__ + r__))

    return right_ascension, dec



def get_altitude_azimuth(utc_time, lon, lat):
    """Returns sun altitude and azimuth from utc_time, lon, and lat.
    """

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    ra_, dec = sun_ra_dec(utc_time)
    h__ = local_hour_angle(utc_time, lon, ra_)
    return (np.arcsin(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h__)),np.arctan2(-np.sin(h__), (np.cos(lat) * np.tan(dec) - np.sin(lat) * np.cos(h__))))



def local_mean_sidereal_time(utc_time, longitude):
    """Returns local mean sidereal time.
    """
    return g_m_sidereal_time(utc_time) + longitude



def cart_2_kep(r_vec,v_vec):
    """Returns keplerian elements from cartesian co-ordinates.
    """

    h_bar = np.cross(r_vec,v_vec)
    h = np.linalg.norm(h_bar)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    E = 0.5*(v**2) - mu/r
    a = -mu/(2*E)
    
    e = np.sqrt(1 - (h**2)/(a*mu))
    i = np.arccos(h_bar[2]/h)

    omega_LAN = np.arctan2(h_bar[0],-h_bar[1])
    lat = np.arctan2(np.divide(r_vec[2],(np.sin(i))),\
    (r_vec[0]*np.cos(omega_LAN) + r_vec[1]*np.sin(omega_LAN)))    
  
    p = a*(1-e**2)
    nu = np.arctan2(np.sqrt(p/mu) * np.dot(r_vec,v_vec), p-r)
  
    omega_AP = lat - nu
   
    EA = 2*np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(nu/2))
    
    n = np.sqrt(mu/(a**3))
    T = t - (1/n)*(EA - e*np.sin(EA))

    return a,e,i,omega_AP,omega_LAN,T, EA



def sun_ecliptic_long(utc_time):
    """Returns ecliptic longitude of sun.
    """
    jdate = days_since_2000(utc_time) / 36525.0

    m_at = np.deg2rad(357.52910 +35999.05030 * jdate -0.0001559 * jdate * jdate - 0.00000048 * jdate * jdate * jdate)

    long_ = 280.46645 + 36000.76983 * jdate + 0.0003032 * jdate * jdate

    temp = ((1.914600 - 0.004817 * jdate - 0.000014 * jdate * jdate) * np.sin(m_at) + (0.019993 - 0.000101 * jdate) * np.sin(2 * m_at) + 0.000290 * np.sin(3 * m_at))
    
    final = long_ + temp
    final_rad = np.deg2rad(final) 

    return final_rad   



def cosine_sun_zenith(utc_time, lon, lat):
    """Cosine of the sun-zenith angle for *lon*, *lat* at *utc_time*.
    """
    lon = np.deg2rad(lon)

    lat = np.deg2rad(lat)
    r_a, dec = sun_ra_dec(utc_time)
    h__ = local_hour_angle(utc_time, lon, r_a)

    final = (np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h__))
    return final
    


def sun_to_earth_distance_correction(utc_time):
    """Calculate the sun earth distance correction, relative to 1 AU, returns the correction value.
    """
    year = 365.256
    correction = 1 - 0.0334 * np.cos(2 * np.pi * (days_since_2000(utc_time) - 2) / year)

    return correction



def keplerian_to_cartesian(a,e,i,omega_AP,omega_LAN,T, EA):
    """Converts keplerian elements to cartesian co-ordinates.
    """    
    a=a*1000
    i=np.deg2rad(i)
    omega_AP=np.deg2rad(omega_AP)
    omega_LAN=np.deg2rad(omega_LAN)
    mu = G.value*M_earth.value
    Re = R_earth.value
    t=0
    n = np.sqrt(mu/(a**3))
    M = n*(t - T)
    
    MA = EA - e*np.sin(EA)


    nu = 2*np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(EA/2))
    
    r = a*(1 - e*np.cos(EA))
    
    h = np.sqrt(mu*a * (1 - e**2))
    
    Om = omega_LAN
    w =  omega_AP

    X = r*(np.cos(Om)*np.cos(w+nu) - np.sin(Om)*np.sin(w+nu)*np.cos(i))
    Y = r*(np.sin(Om)*np.cos(w+nu) + np.cos(Om)*np.sin(w+nu)*np.cos(i))
    Z = r*(np.sin(i)*np.sin(w+nu))

    
    p = a*(1-e**2)

    V_X = (X*h*e/(r*p))*np.sin(nu) - (h/r)*(np.cos(Om)*np.sin(w+nu) + \
    np.sin(Om)*np.cos(w+nu)*np.cos(i))
    V_Y = (Y*h*e/(r*p))*np.sin(nu) - (h/r)*(np.sin(Om)*np.sin(w+nu) - \
    np.cos(Om)*np.cos(w+nu)*np.cos(i))
    V_Z = (Z*h*e/(r*p))*np.sin(nu) - (h/r)*(np.cos(w+nu)*np.sin(i))

    return [X/1000,Y/1000,Z/1000,V_X/1000,V_Y/1000,V_Z/1000]    



def position_ECI(time, lon, lat, alt):
    """Calculate and returns observer ECI position.
    """

    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)

    theta = (g_m_sidereal_time(time) + lon) % (2 * np.pi)
    c = 1 / np.sqrt(1 + F * (F - 2) * np.sin(lat)**2)
    sq = c * (1 - F)**2

    achcp = (rad_earth * c + alt) * np.cos(lat)
    x = achcp * np.cos(theta)  # kilometers
    y = achcp * np.sin(theta)
    z = (rad_earth * sq + alt) * np.sin(lat)

    vx = -M_fact * y  # kilometers/second
    vy = M_fact * x
    vz = 0

    return (x, y, z), (vx, vy, vz)



