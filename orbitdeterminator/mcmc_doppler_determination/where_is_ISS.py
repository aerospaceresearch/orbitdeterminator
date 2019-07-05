import numpy as np
import matplotlib.pyplot as plt

import astropy.coordinates as coord
import astropy.units as u
from astropy import time

import poliastro
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

import urllib.request
import time as time_o


# Download TLE of many Stations.
url = 'http://www.celestrak.com/NORAD/elements/stations.txt'
response = urllib.request.urlopen(url)
data = response.read()      
text = data.decode('utf-8') 
data = text.splitlines()

# Extract data of "ISS (ZARYA)"
for i in range(len(data)):
    if data[i].startswith('ISS (ZARYA)'):
        
        l1 = data[i+1].split()

        time_vector = l1[3]        
        year = int("20"+time_vector[0:2])
        ydayfraction = float(time_vector[2:])
        
        
        
        l2 = data[i+2].split()
        
        Inclination    = float(l2[2]) * u.deg
        RA_of_node     = float(l2[3]) * u.deg
        Eccentricity   = float("0."+l2[4]) * u.one
        Arg_of_perigee = float(l2[5]) * u.deg
        Mean_anomaly   = float(l2[6]) * u.deg
        Rev_Per_Day    = (float(l2[7])  / u.day).to(1/u.s)
        semi_major_Axis= Earth.k**(1/3) / (2 * np.pi * Rev_Per_Day)**(2/3)

Epoch_time = time.Time("{}-01-01 00:00".format(year))
Epoch_time += (ydayfraction -1) * u.day

message = ("\tInclination    : {}\n"
           "\tRA of node     : {}\n"
           "\tEccentricity   : {}\n"
           "\tArg of Perigee : {}\n"
           "\tMean_Anomaly   : {}\n"
           "\tSemi-major Axis: {}\n\n"
           "\tEpoch          : {}").format(Inclination, 
                                           RA_of_node, 
                                           Eccentricity,
                                           Arg_of_perigee, 
                                           Mean_anomaly, 
                                           semi_major_Axis,
                                           Epoch_time.yday)

print("Linux: Press ctr + C to kill this task.\n\n")

print("ISS Orbital Elements from TLE\n\n" + message ) 

# Generate the poliastro Orbit object from Classical Elements.
ZAYRA = poliastro.twobody.Orbit.from_classical(Earth,
                                              semi_major_Axis,
                                              Eccentricity,
                                              Inclination,
                                              RA_of_node,
                                              Arg_of_perigee,
                                              Mean_anomaly,
                                              Epoch_time)

while(True):
    Tnow = time.Time.now()
    DeltaT = Tnow - Epoch_time

    # Do the time evolution from Epoch to Tnow
    ISSnow = ZAYRA.propagate( DeltaT )

    # Transform ISS space coordinates from the inertial system (GCRS) to
    # the common lat-long (ITRS)
    ISS_gcrs = coord.GCRS(
    coord.CartesianRepresentation(x=ISSnow.r[0], y=ISSnow.r[1], z=ISSnow.r[2]),
    obstime=time.Time.now())

    ISS_itrs = ISS_gcrs.transform_to(coord.ITRS(obstime=Tnow))
    ISS_el = coord.EarthLocation(ISS_itrs.x, ISS_itrs.y, ISS_itrs.z)
    lon, lat, height = ISS_el.to_geodetic()


    message = ("\n\nISS coordinates now:"
          "\n\tTime        : {}"
          "\n\tLatitude    : {}"
          "\n\tLongitude   : {}"
          "\n\tAltitude    : {}").format(Tnow,
                                         lat.deg, 
                                         lon.deg,
                                         height )

    print(message)
    time_o.sleep(2)