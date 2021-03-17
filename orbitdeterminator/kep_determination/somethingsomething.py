from astropy.coordinates import EarthLocation
from astropy.coordinates import Angle
from astropy.time import Time
from astropy import units as uts
from datetime import datetime
import math

import numpy

observing_location = EarthLocation(lat=46.57*uts.deg, lon=7.65*uts.deg)
observing_time = Time(datetime.utcnow(), scale='utc', location=observing_location)
lst = observing_time.sidereal_time('mean')
#print(observing_location)
a = Angle(lst)
print(a.degree)
#print(datetime.utcnow())

def equatorial_to_horizon(al, az, lat, lon, utc):
        AL = math.radians(al)
        AZ = math.radians(az)
        LAT = math.radians(lat)
        #converting utc time to local siderial time in radians
        observing_location = EarthLocation(lat=lat*uts.deg, lon=lon*uts.deg)
        observing_time = Time(utc, scale='utc', location=observing_location)
        ST = Angle(observing_time.sidereal_time('mean'))
        ST = ST.degree
        
        sin_dec = (math.sin(LAT)*math.sin(AL)) + (math.cos(LAT)*math.cos(AL)*math.cos(AZ))
        dec = math.degrees(math.asin(sin_dec))

        cos_HA = (math.sin(AL) - math.sin(LAT)*sin_dec)/(math.cos(LAT)*math.cos(math.asin(sin_dec)))
        HA = math.degrees(math.acos(cos_HA))
        #if the object is west of observer's meridian
        if(ST-HA > 0):
            ra = ST - HA
        #if the object is east of observer's meridian
        else:
            ra = ST + HA

        return ra, dec

RA, DEC = equatorial_to_horizon(2073, 22.5, 38.9478, -104.5614, datetime.utcnow())
print(RA, DEC)
print(datetime.utcnow())