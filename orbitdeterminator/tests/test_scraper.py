import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pytest
import MySQLdb
from numpy.testing import assert_array_equal
from database.scraper import database_connect, string_to_hash

def test_connection():
    _,connect = database_connect()
    assert(connect == 0)

"""
The satellite name in the TLE is max 24 character in length. Hence, the below
function is taking it with the rest whitespaces
"""
def test_hash():
    md5 = string_to_hash("SWISSCUBE               ")
    assert(md5 == "3ca5eaf04b8badf4564ef8f753c56e12")

    md5 = string_to_hash("EXOCUBE                 ")
    assert(md5 == "8559e7a3093811263cd8b251f7bac5b7")

if __name__ == "__main__":
    test_connection()
    test_hash()
