"""
This code creates database and tables to each satellite with their md5 hash as table name.

There are 2 codes for database: init_database.py and scraper.py

Important: Run init_database.py before scraper.py

Note: Password filed is empty during database connection. Insert password for
      mysql authentication.
"""

import hashlib
import MySQLdb
import requests
from bs4 import BeautifulSoup

page = requests.get("https://www.celestrak.com/NORAD/elements/cubesat.txt")
soup = BeautifulSoup(page.content, 'html.parser')
tle = list(soup.children)
tle = tle[0].splitlines()

"""
Creating database named "cubesat"
"""

db = MySQLdb.connect(host="localhost", user="root", passwd="mysql")
cursor = db.cursor()
sql = 'CREATE DATABASE cubesat;'
cursor.execute(sql)
print('Database created')
cursor.execute('use cubesat;')
print('Database selected')

"""
Creating tables with md5 hash of satellite name as table name
"""

sql = 'CREATE TABLE mapping ' + '\
(sat_name varchar(50), md5_hash varchar(50));'
cursor.execute(sql)

success = 0
error = 0
for i in range(0, len(tle), 3):
    md5_hash = hashlib.md5(tle[i].encode())
    sat_hash = md5_hash.hexdigest()

    try:
        sql = 'CREATE TABLE ' + str(sat_hash) + '\
        (time varchar(30), line1 varchar(70), line2 varchar(70));'
        cursor.execute(sql)
    except Exception:
        error = error + 1
        print(tle[i] + ' - Error: Table not created')
    else:
        sql = 'INSERT INTO mapping values(\'%s,\', \'%s\');\
        ' %(str(tle[i]), str(sat_hash))
        cursor.execute(sql)
        db.commit()
        success = success + 1

print('Tables created : ' + str(success))
# print('Error/Total : ' + str(error) + '/' + str(error+success))
db.close()
