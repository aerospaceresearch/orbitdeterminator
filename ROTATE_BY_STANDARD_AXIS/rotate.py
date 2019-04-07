'''
runs in python 2.7
To run in python 3.x remove fields and the line fields=crd.next()
'''
import csv
from PySide.QtGui import QApplication,QWidget,QComboBox,QVBoxLayout,QLabel,QFormLayout
import os
import sys
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
'''
This program takes an input .csv file containing position vectors in format(time,x,y,z) and a .txt file
containing any angle(in degree) and rotates the figure.
Application-can restrict the problem in x-y plane. 
'''
def rotateByYAxis(ideg,r):
    '''
    Takes a degree and position vectors and rotate the whole figure by x-axis,y-axis,z-axis and plots it.
    
    Args:
        ideg(float)=angle to be rotated(in degree).
        r(numpy array)=array of position vectors of all points(excluding time)
    '''
    irad=np.radians(ideg)
    Rx=np.asarray([[1,0,0],[0,np.cos(irad),-np.sin(irad)],[0,np.sin(irad),np.cos(irad)]])
    Ry=np.asarray([[np.cos(irad),0,np.sin(irad)],[0,1,0],[-np.sin(ideg),0,np.cos(ideg)]])
    Rz=np.asarray([[np.cos(irad),-np.sin(irad),0],[np.sin(irad),np.cos(irad),0],[0,0,1]])
    rvx=np.matmul(Rx,r)
    rvy=np.matmul(Ry,r)
    rvz=np.matmul(Rz,r)
    print("After rotation")
    #print(rv)
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot3D(rvx[0,:],rvx[1,:],rvx[2,:],label="rotation x")
    ax.plot3D(rvy[0,:],rvy[1,:],rvy[2,:],label="rotation y")
    ax.plot3D(rvz[0,:],rvz[1,:],rvz[2,:],label="rotation z")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    

def readFile(CSVfilename,textfilename):
    '''
        Reads data from files and stores them in required data structure
        args:
            CSVfilename(String):name of .csv file
            textfilename(String):name of text file containing angle
    '''
    r=[]
    with open(CSVfilename,'r') as csvfile:
        crd=csv.reader(csvfile)
        fields=crd.next()
        for rows in crd:
            if("\t" in rows[0]):
                L=list(map(float,rows[0].split("\t")))
            else:
                print(rows)
                L=list(map(float,rows))
            L.pop(0)
            r.append(L)
        #print(r) 
        p=np.asarray(r)
        pt=np.ndarray.transpose(p)
        #print(pt) 
        fp=open(textfilename,"r")
        line=fp.read()
        inc=map(float,line.split())
        #print (inc)
        ia=(inc[0]+inc[1])/2
        rotateByYAxis(ia,pt)



class SearchFile(QWidget):
    
    def __init__(self):
        '''
            Creates GUI
        '''
        QWidget.__init__(self)
        self.setWindowTitle("Search window")
        self.fn1=""
        self.fn2=""
        L1=[]
        L2=[]
        st=os.getcwd()
        L=os.listdir(st)
        for filenames in L:
            if(".txt" in filenames):
                L1.append(filenames)
            if(".csv" in filenames):
                L2.append(filenames)
        self.files1=QComboBox() 
        self.files2=QComboBox() 
        #self.files1.setText("SatelliteDataFile")
        #print(self.files1)
        #print(self.files2)
        self.files1.addItems(L1)
        self.files1.setCurrentIndex(-1)
        self.files2=QComboBox() 
        self.files2.addItems(L2)
        self.files2.setCurrentIndex(-1) 
        self.files1.currentIndexChanged.connect(lambda:self.returnString(self.files1))
    
        self.files2.currentIndexChanged.connect(lambda:self.returnString(self.files2))
        self.setUpUI()
        
        
    def setUpUI(self):
        layout=QVBoxLayout()
        form=QFormLayout()
        form.addRow(QLabel("text file"),self.files1)  
        form.addRow(QLabel(".csv file"),self.files2) 
        layout.addLayout(form)             
        self.setLayout(layout)
        
    def returnString(self,files):
        #print(files)
        if(files==self.files1):
            self.fn1=str(self.files1.currentText())
        else:
            self.fn2=str(self.files2.currentText())
        #print(self.fn1)
        #print(self.fn2)
        

app=QApplication(sys.argv)   
sf=SearchFile()
sf.show()
#print(sf.files1.currentIndexChanged.connect(lambda:sf.returnString(sf.files1)))
(app.exec_())
readFile(sf.fn2,sf.fn1)
