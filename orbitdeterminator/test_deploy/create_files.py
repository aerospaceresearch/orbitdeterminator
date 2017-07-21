'''
Create a new .txt files every second
'''
import time
count = 0
while True:
    a = open("src/file_%s.txt" % (count), 'w')
    a.write("this is file" + str(count))
    count += 1
    time.sleep(1)
