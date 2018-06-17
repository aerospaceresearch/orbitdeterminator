import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pytest
import MySQLdb
from numpy.testing import assert_array_equal
from database.initDatabase import create_database, string_to_hash, create_table

def test_create_database():
    db, flag = create_database("test_database0")
    assert(flag == 0)

    return db

"""
The satellite name in the TLE is max 24 character in length. Hence, the below
function is taking it with the rest whitespaces
"""
def test_string_to_hash():
    md5 = string_to_hash("SWISSCUBE               ")
    assert(md5 == "3ca5eaf04b8badf4564ef8f753c56e12")

    md5 = string_to_hash("EXOCUBE                 ")
    assert(md5 == "8559e7a3093811263cd8b251f7bac5b7")

    md5 = string_to_hash("BRAC ONNESHA            ")
    assert(md5 == "3e1e0c4e60e927dfeb8f3dbf77929358")
    # print("1 " + str(md5))

    md5 = string_to_hash("@@@@")
    assert(md5 == "711e29f67094d95da05270e362bc4c83")
    # print("2 " + str(md5))

    md5 = string_to_hash("1234")
    assert(md5 == "81dc9bdb52d04dc20036dbd8313ed055")
    # print("3 " + str(md5))

    md5 = string_to_hash("table123")
    assert(md5 == "c3c08f07d2f6e853e87811899e36a27e")
    # print("4 " + str(md5))

    md5 = string_to_hash("123table")
    assert(md5 == "b8633a05d10680817a7fad9e41d6cbb0")
    # print("5 " + str(md5))

    md5 = string_to_hash("table-123")
    assert(md5 == "5770d2f18119e6f2d92dc54e19314f06")
    # print("6 " + str(md5))

    md5 = string_to_hash("table_123")
    assert(md5 == "6d74287e5b81a95fa018bc05bb9d188e")
    # print("7 " + str(md5))

    md5 = string_to_hash("cute 123")
    assert(md5 == "7fb7f792cffc2794ec217ea30d4cd313")
    # print("8 " + str(md5))

    md5 = string_to_hash("cute (123)")
    assert(md5 == "66265856f35ad1e7a0c37ec714ce5175")
    # print("9 " + str(md5))

    md5 = string_to_hash("cute-(123)")
    assert(md5 == "fc292325f1d87e3fc4795f99e500b9d0")
    # print("10 " + str(md5))

    md5 = string_to_hash("cute_(123)")
    assert(md5 == "2dc586ab6a6d94582fd1df06a8a7c7e9")
    # print("11 " + str(md5))

    md5 = string_to_hash("cute,123")
    assert(md5 == "8c08b27d843ae3831fbd62e35a656df9")
    # print("12 " + str(md5))

def test_create_table(db):
    flag = create_table(db, "BRAC ONNESHA            ")
    # print(flag)
    assert(flag == 1)

    flag = create_table(db, "@@@@")
    # print(flag)
    assert(flag == 1)

    flag = create_table(db, "1234")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "table123")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "123table")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "table-123")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "table_123")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "cute 123")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "cute (123)")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "cute-(123)")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "cute_(123)")
    # print(flag)
    assert(flag == 0)

    flag = create_table(db, "cute,123")
    # print(flag)
    assert(flag == 0)

if __name__ == "__main__":
    db = test_create_database()
    test_string_to_hash()
    test_create_table(db)
