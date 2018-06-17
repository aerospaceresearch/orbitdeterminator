"""
This code populates the database with tle entries.

There are 2 codes for database: init_database.py and scraper.py

Important: Run init_database.py before scraper.py

Note: Password filed is empty during database connection. Insert password for
      mysql authentication.
"""

import time
import hashlib
import MySQLdb
import requests
from bs4 import BeautifulSoup

def database_connect(name):
    """
    Initializes database connection and selects corresponding database

    Args:
        name : database name

    Return:
        db : database object
        d : connection flag (0: success)
    """

    db = MySQLdb.connect(host="localhost", user="root", passwd="mysql")
    cursor = db.cursor()
    d = cursor.execute('use ' + name)
    # print('Database selected')
    return db, d

def string_to_hash(tle):
    """
    Converts satellite name to its corresponding md5 hexadecimal hash

    Args:
        tle : satellite name

    Return:
        sat_hash : md5 hexadecimal hash
    """

    md5_hash = hashlib.md5(tle.encode())
    sat_hash = md5_hash.hexdigest()
    return sat_hash

def update_table(db, line0, line1, line2):
    """
    Updating tables with new TLE values.

    There are 3 attributes in the tables are: time, line1, line2

    Args:
        db : database object
        line0 : satellite name
        line1 : line 1 of TLE
        line2 : line2 of TLE

    Return:
        d : flag variable for table update (None : success)
    """

    ts = time.time()
    cursor = db.cursor()

    sat_hash = string_to_hash(line0)
    try:
        sql = 'INSERT INTO %s values(\'%s,\', \'%s,\', \'%s,\');\
        ' %(str(sat_hash), str(ts), line1, line2)
        cursor.execute(sql)
        d = db.commit()
    except Exception:
        d = 1
        # print(line0 + ' - Error: Table not found')
    # else:
    #     print(d)

    return d

def scrap_data(db):
    """
    Scrapes data from celestrak site and calls update_table() to update the respective tables in the database.

    Args:
        db : database object

    Returns:
        NIL
    """

    page = requests.get("https://www.celestrak.com/NORAD/elements/cubesat.txt")
    soup = BeautifulSoup(page.content, 'html.parser')
    tle = list(soup.children)
    tle = tle[0].splitlines()

    success = 0
    error = 0
    for i in range(0, len(tle), 3):
        d = update_table(db, tle[i], tle[i+1], tle[i+2])
        if(d == None):
            success += 1
        else:
            error += 1

    # print('Tables updated : ' + str(success))
    # print('Error/Total : ' + str(error) + '/' + str(error+success))
    db.close()

if __name__ == "__main__":
    db,_ = database_connect("cubesat")
    scrap_data(db)
