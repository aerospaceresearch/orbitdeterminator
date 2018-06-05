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

page = requests.get("https://www.celestrak.com/NORAD/elements/cubesat.txt")
soup = BeautifulSoup(page.content, 'html.parser')
tle = list(soup.children)
tle = tle[0].splitlines()
ts = time.time()

db = MySQLdb.connect(host="localhost", user="root", passwd="*****")
cursor = db.cursor()
cursor.execute('use cubesat;')
print('Database selected')

"""
Updating tables with new TLE values.

There are 3 attributes in the tables are: time, line1, line2
"""
success = 0
error = 0
for i in range(0, len(tle), 3):
    md5_hash = hashlib.md5(tle[i].encode())
    sat_hash = md5_hash.hexdigest()

    try:
        sql = 'INSERT INTO %s values(\'%s,\', \'%s,\', \'%s,\');\
        ' %(str(sat_hash), str(ts), tle[i+1], tle[i+2])
        cursor.execute(sql)
        db.commit()
    except Exception:
        error = error + 1
        # print(tle[i] + ' - Error: Table not found')
    else:
        success = success + 1

print('Tables updated : ' + str(success))
# print('Error/Total : ' + str(error) + '/' + str(error+success))
db.close()
