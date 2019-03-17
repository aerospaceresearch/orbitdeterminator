import mysql.connector
import sys
import csv
'''
stores a .csv file in remote database so that they can be used by multiple workstations.
Create a new database on online server and insert the details in the file named inputDetails.txt in the format mentioned
'''
def insertIntoTable(tableName,datafile,host,databaseName,userName,Password=None):
    '''
        args-tableName(string):name odf the table
        datafile:.csv file to be stored in database
        host:name of the online database host
        databaseName:name of the database(or name of the satellite)
        userName:username for the database
        Password:Password for the database
    '''
    mydb=mysql.connector.connect(host=host,user=userName,passwd=Password,database=databaseName)
    cursor=mydb.cursor()
    i=1
    with open(datafile,"r") as csvfile:
        cr=csv.reader(csvfile)
        for row in csvfile:
            if(',' in row):
                c=','
            else:
                c='\t'
            time,x,y,z=row.split(c)
            cursor.execute("insert into "+tableName+" values (%s,%s,%s,%s,%s)",(satelliteid,time,x,y,z))
            mydb.commit()
            print("Entered ",i,"th term successfully")
            i=i+1
            
    

def createTable(satelliteid,tableName,datafile,host,databaseName,userName,Password=None):
    '''
        args-tableName(string):name odf the table
        datafile:.csv file to be stored in database
        host:name of the online database host
        databaseName:name of the database(or name of the satellite)
        userName:username for the database
        Password:Password for the database
    '''
    mydb=mysql.connector.connect(host=host,user=userName,passwd=Password,database=databaseName)
    cursor=mydb.cursor()
    print("Successfully created table Satellite1")
    cursor.execute("SELECT * FROM information_schema.TABLES where table_type='base table'")
    tableList=cursor.fetchall()
    for i in range(len(tableList)):
        tableList[i]=tableList[i][2]
    flag=0
    for i in range(len(tableList)):
        if(tableList[i]==tableName):
            flag=1
            break
            
    if(flag==0):
        cursor.execute("create table "+tableName
                        +'''(id varchar(30),
                           time varchar(30),
                           x varchar(30),
                           y varchar(30),
                           z varchar(30));
                          ''')
        
    insertIntoTable(tableName,datafile,host,databaseName,userName,Password)
    
    
    
    
if __name__=='__main__':
    
    f=open("inputDetails.txt","r")
    '''
    inputDetails.txt contain satellite_id,satellite_name,.csv_data_file_name,user_name,password_for_database,database_name
    separated by space
    '''
    line=f.read()
    #print(line)
    satelliteid,tableName,datafile,host,userName,Password,databaseName=line.split()
    createTable(satelliteid=satelliteid,
                tableName=tableName,
                datafile=datafile,
                host=host,
                userName=userName,
                Password=Password,
                databaseName=databaseName
                )
                
                
