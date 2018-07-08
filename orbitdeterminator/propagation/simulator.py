import sys
import time
import signal
import threading
from functools import partial
import numpy as np

from orbitdeterminator.propagation.cowell import propagate_state
from orbitdeterminator.util.teme_to_ecef import conv_to_ecef
from orbitdeterminator.util.new_tle_kep_state import kep_to_state

class Simulator():

    def __init__(self,params):

        self.s         = kep_to_state(params.kep).flatten()
        self.t0        = params.epoch
        self.t         = params.t0-params.period
        self.period    = params.period
        self.speed     = params.speed
        self.op_writer = params.op_writer

        self.s = propagate_state(self.s,self.t0,self.t)
        self.t0 = self.t

        self.calc_thr = None
        self.is_running = False

    def simulate(self):
        self.is_running = True

        self.op_writer.open()
        self.calc_thr = threading.Timer(0, self.calc)
        self.calc_thr.start()

        # listen for commands.
        # only quit command implemented for now
        while self.is_running:
            c = input()
            if (c == 'q'):
                self.stop()

    def calc(self):
        calc_period = max(0,self.period/self.speed)
        self.calc_thr = threading.Timer(calc_period, self.calc)
        self.calc_thr.start()
        self.t += self.period
        self.s = propagate_state(self.s,self.t0,self.t)
        self.t0 = self.t
        self.op_writer.write(self.t,self.s)

    def stop(self):
        if self.calc_thr is not None:
            self.calc_thr.cancel()
        self.is_running = False
        self.op_writer.close()

def sig_handler(simulator, signal, frame):
    simulator.stop()
    sys.exit(0)

class OpWriter():
    def open(self):
        pass

    @staticmethod
    def write(t,s):
        print(t,*s)

    def close(self):
        pass

class print_r(OpWriter):
    @staticmethod
    def write(t,s):
        print(t,*s[0:3])

class print_lat_lon(OpWriter):
    @staticmethod
    def write(t,s):
        t,lat,lon,alt = conv_to_ecef(np.array([[t,*s[0:3]]]))[0]
        print(t,lat,lon,alt)

class save_r(OpWriter):
    def __init__(self, name):
        self.file_name = name

    def open(self):
        #self.f = open(self.file_name,'a')
        self.t = None
        self.iter = 0
        self.f = open(self.file_name,'a')
        self.f.write('# Begin write\r\n')
        self.f.close()

    def write(self,t,s):
        if not self.t == t:
            self.f = open(self.file_name,'a')
            self.f.write("{} {} {} {} {} {} {}\r\n".format(t,*s))
            self.f.close()
            self.t = t
            print("\rIteration:",self.iter,end=' '*10)
            self.iter+=1

    def close(self):
        self.f.close()

class SimParams():
    kep = None
    epoch = None
    period = 1
    t0 = int(time.time())
    speed = 1
    op_writer = OpWriter()

if __name__ == "__main__":
    epoch = 1531152114
    #epoch = 1530729961
    #iss_kep = np.array([6775,0.0002893,51.6382,211.1340,7.1114,148.9642])
    #iss_kep = np.array([6786.6787,0.0003411,51.6428,263.9950,291.0075,245.8091])
    iss_kep = np.array([6786.6420,0.0003456,51.6418,290.0933,266.6543,212.4306])
    #6783.1714
    #6786.5714

    params = SimParams()
    params.kep = iss_kep
    params.epoch = epoch
    params.op_writer = print_lat_lon() #save_r('ISS.csv')

    s = Simulator(params)
    signal.signal(signal.SIGINT, partial(sig_handler,s))
    s.simulate()
