import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal
from kep_determination.sgp4 import *

def test_find_year():
    obj = SGP4()

    year = 18
    year = obj.find_year(year)
    ans = 2018
    assert(year == ans)

    year = 72
    year = obj.find_year(year)
    ans = 1972
    assert(year == ans)

    del(obj)

def test_find_date():
    obj = SGP4()

    date = '18194'
    date = obj.find_date(date)
    ans = (2018, 7, 14)
    assert(date == ans)

    date = '72193'
    date = obj.find_date(date)
    ans = (1972, 7, 12)
    assert(date == ans)

    del(obj)

def test_find_time():
    obj = SGP4()

    time = 6084328
    time = obj.find_time(time)
    ans = (7, 52, 19)
    assert(time == ans)

    time = 78026175
    time = obj.find_time(time)
    ans = (4, 12, 0)
    assert(time == ans)

    del(obj)

def test_julian_day():
    obj = SGP4()

    year, mon, day, hr, mts, sec = 2018, 7, 13, 16, 48, 35
    jd = obj.julian_day(year, mon, day, hr, mts, sec)
    jd = float("{0:.5f}".format(jd))
    ans = 2458313.20041
    assert_almost_equal(jd, ans)

    year, mon, day, hr, mts, sec = 2017, 12, 31, 23, 59, 59
    jd = obj.julian_day(year, mon, day, hr, mts, sec)
    jd = float("{0:.5f}".format(jd))
    ans = 2458119.49999
    assert_almost_equal(jd, ans)

    year, mon, day, hr, mts, sec = 2018, 1, 1, 00, 00, 00
    jd = obj.julian_day(year, mon, day, hr, mts, sec)
    jd = float("{0:.5f}".format(jd))
    ans = 2458119.50000
    assert_almost_equal(jd, ans)

    del(obj)

def test_update_epoch():
    obj = SGP4()

    year, mon, day, hr, mts, sec = 2018, 7, 13, 16, 59, 59
    epoch = obj.update_epoch(year, mon, day, hr, mts, sec)
    ans = (2018, 7, 13, 17, 0, 0)
    assert(epoch == ans)

    year, mon, day, hr, mts, sec = 2017, 12, 31, 23, 59, 59
    epoch = obj.update_epoch(year, mon, day, hr, mts, sec)
    ans = (2018, 1, 1, 0, 0, 0)
    assert(epoch == ans)

    year, mon, day, hr, mts, sec = 2018, 1, 1, 00, 00, 00
    epoch = obj.update_epoch(year, mon, day, hr, mts, sec)
    ans = (2018, 1, 1, 0, 0, 1)
    assert(epoch == ans)

    del(obj)

if __name__ == "__main__":
    test_find_year()
    test_find_date()
    test_find_time()
    test_julian_day()
    test_update_epoch()
