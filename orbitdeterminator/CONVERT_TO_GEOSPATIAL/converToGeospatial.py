'''
runs in python 2.7
This program converts topocentric data to geocentric data.
'''
from PySide.QtGui import QApplication,QWidget,QComboBox,QVBoxLayout,QLabel,QFormLayout
import os
import csv
import sys
topo=[]
sp=[]
wp=[]
geo=[]
i=0
filename1=""
filename2=""
def writeToFile(filename3,geo):
    '''
    writes the data to a .csv file.
    args:
        filename3(String):name of data file where data is to be stored
        geo(list):Data to be stored in format(time,x,y,z)
    '''
    fields=["time","x","y","z"]
    #print(geo)
    with open(filename3,'w') as filewriter:
        csvwriter=csv.writer(filewriter)
        csvwriter.writerow(fields)
        csvwriter.writerows(geo)
                

def readFromFile(filename1,filename2):
    '''
    reads data from 2 files and converts the topocentric coordinate-system to geocentric system
    args:
        filename1(String):topocentric position data file(.csv file)
        filename2(String):file containing position of workstation..txt if single workstation is concerned
                          .csv if multiple workstations are concerned.  
    '''
    filename3="g"+filename1
    if(".txt" in filename2):
        fp=open(filename2,"r")
        line=fp.read()
        wp=map(float,line.split())
        i=0
        workstationPosition=list(wp)
        with open(filename1, 'r') as csvfile: 
            csvreader = csv.reader(csvfile) 
            for row in csvreader:
                gs=[]
                i=i+1
                sp=map(float,row[0].split("\t"))
                
                satellitePosition=list(sp)
                
                gs.append(satellitePosition[0])
                gs.append(satellitePosition[1]+workstationPosition[0])
                gs.append(satellitePosition[2]+workstationPosition[1])
                gs.append(satellitePosition[3]+workstationPosition[2])
                geo.append(gs)
                
            writeToFile(filename3,geo)
            print("Added ",i," data successfully")
      
                
    if(".csv" in filename2):
        with open(filename1,'r') as fp1:
            cr1=csv.reader(fp1)
            sp=[]
            for r1 in cr1:
                
                sp.append(list(map(float,r1[0].split("\t"))))
            
        with open(filename1,'r') as fp2:
            cr2=csv.reader(fp2)
            wp=[]
            for r2 in cr2:
               
                wp.append(list(map(float,r2[0].split("\t"))))
                
        if(len(wp)!=len(sp)):
            print(".csv files have different lengths")
        else:
            for i in range(len(wp)):
                gs=[]
                gs.append(sp[i][0])
                gs.append(sp[i][1]+wp[i][0])
                gs.append(sp[i][2]+wp[i][1])
                gs.append(sp[i][3]+wp[i][2])
                geo.append(gs)
            writeToFile(filename3,geo)
            print("Added ",i,"Data Successfully")
                
                
class SearchFile(QWidget):
    def __init__(self):
        'Makes GUI'
        QWidget.__init__(self)
        self.setWindowTitle("Search window")
        self.fn1=""
        self.fn2=""
        L1=[]
        st=os.getcwd()
        L=os.listdir(st)
        for filenames in L:
            if(".txt" in filenames or ".csv" in filenames):
                L1.append(filenames)
        self.files1=QComboBox() 
        self.files2=QComboBox() 
        #self.files1.setText("SatelliteDataFile")
        #print(self.files1)
        #print(self.files2)
        self.files1.addItems(L1)
        self.files1.setCurrentIndex(-1)
        self.files2=QComboBox() 
        self.files2.addItems(L1)
        self.files2.setCurrentIndex(-1) 
        self.files1.currentIndexChanged.connect(lambda:self.returnString(self.files1))
    
        self.files2.currentIndexChanged.connect(lambda:self.returnString(self.files2))
        self.setUpUI()
        
        
    def setUpUI(self):
        layout=QVBoxLayout()
        form=QFormLayout()
        form.addRow(QLabel("Satellite Data File"),self.files1)  
        form.addRow(QLabel("Work Station Position DataFile"),self.files2) 
        layout.addLayout(form)             
        self.setLayout(layout)
        
    def returnString(self,files):
        
        if(files==self.files1):
            self.fn1=str(self.files1.currentText())
        else:
            self.fn2=str(self.files2.currentText())
        
        

app=QApplication(sys.argv)   
sf=SearchFile()
sf.show()
(app.exec_())
readFromFile(sf.fn1,sf.fn2)


