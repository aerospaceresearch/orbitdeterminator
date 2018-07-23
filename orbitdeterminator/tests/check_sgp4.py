import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv
from kep_determination.sgp4 import *

def update_epoch(yr, mth, day, hr, mts, sec):
    '''
    Adds one second to the given time.

    Args:
        self: class variables
        yr: year
        mth: month
        day: date
        hr: hour
        mts: minutes
        sec: seconds

    Returns:
        updated timestamp epoch in a tuple with value as (year, month, day,
        hour, minute, second)
    '''
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

if __name__ == "__main__":
    line1 = "1 88888U          80275.98708465  .00073094  13844-3  66816-4 0     8"
    line2 = "2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518   105"

    s = twoline2rv(line1,line2,wgs72)
    i = 0
    yr,mth,day,hr,mts,sec = 2018,7,21,19,28,41
    while(i < 1):
        pos,vel = s.propagate(yr,mth,day,hr,mts,sec)
        data = [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]
        data = [float("{0:.5f}".format(i)) for i in data]
        print(str(i) + " - " + str(hr) + ":" + str(mts) + ":" + str(sec) + " - " + str(data))
        yr, mth, day, hr, mts, sec = update_epoch(yr, mth, day, hr, mts, sec)
        i = i + 1
