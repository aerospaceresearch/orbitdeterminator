"""
This code creates database and tables to each satellite with their md5 hash as table name.

There are 2 codes for database: init_database.py and scraper.py

Important: Run init_database.py before scraper.py

Note: Password field is empty during database connection. Insert password for
      mysql authentication.
"""

import hashlib
import mysql.connector as mc
import requests
from bs4 import BeautifulSoup

def create_database(name):
    """
    Creating database named "cubesat"

    Args:
        name : database name

    Return:
        db : database object
        d : connection flag (0: success)
    """

    db = mc.connect(user='root', password='mysql', host='localhost')
    cursor = db.cursor()
    sql = 'CREATE DATABASE ' + name + ';'
    cursor.execute(sql)
    # print('Database created')
    sql = 'use ' + name
    d = cursor.execute(sql)
    # print('Database selected')
    # print(d)
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

def create_table(db, name):
    """
    Creating tables in the database

    Args:
        db : database object
        name : table name

    Return:
        d : flag variable for table update (0 : success / 1 : error)
    """

    cursor = db.cursor()
    sat_hash = string_to_hash(name)

    try:
        sql = 'CREATE TABLE ' + str(sat_hash) + '\
        (time varchar(30), line1 varchar(70), line2 varchar(70));'
        d = cursor.execute(sql)
    except Exception:
        d = 1
        # print("Error: Table not created, " + name)

    return d

def scrap_data(db):
    """
    Scrapes data from celestrak site and calls create_table() to create the respective tables in the database.

    Args:
        db : database object

    Returns:
        NIL
    """

    page = requests.get("https://www.celestrak.com/NORAD/elements/cubesat.txt")
    soup = BeautifulSoup(page.content, 'html.parser')
    tle = list(soup.children)
    tle = tle[0].splitlines()

    cursor = db.cursor()
    sql = 'CREATE TABLE mapping ' + '\
    (sat_name varchar(50), md5_hash varchar(50));'
    d = cursor.execute(sql)

    success = 0
    error = 0
    for i in range(0, len(tle), 3):
        d = create_table(db, tle[i])
        if(d == 0):
            sql = 'INSERT INTO mapping values(\'%s,\', \'%s\');\
            ' %(str(name), str(sat_hash))
            cursor.execute(sql)
            db.commit()
            success += 1
        else:
            error += 1

    # print('Tables created : ' + str(success))
    # print('Error/Total : ' + str(error) + '/' + str(error+success))
    db.close()

if __name__ == "__main__":
    db,_ = create_database("d1")
    scrap_data(db)
