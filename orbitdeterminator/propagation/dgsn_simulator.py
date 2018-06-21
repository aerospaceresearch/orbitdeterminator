import sys
import time
import signal
import threading
from functools import partial
import numpy as np

from orbitdeterminator.propagation.sgp4_prop import propagate
from orbitdeterminator.util.teme_to_ecef import conv_to_ecef

class Simulator():

    def __init__(self,kep,epoch,period,t0=int(time.time()),speed=1,op_period=None,op_func=None):
        if op_period is None:
            op_period = max(1,int(period/speed))
        if op_func is None:
            op_func = self._print

        self.kep = kep
        self.epoch = epoch
        self.t = t0-period
        self.period = period
        self.speed = speed
        self.op_period = op_period
        self.op_func = op_func

        self.r = None
        self.v = None

        self.calc_thr = None
        self.op_thr = None

        self.is_running = False
        
    def simulate(self):
        self.is_running = True

        self.calc_thr = threading.Timer(0, self.calc)
        self.calc_thr.start()
        self.op_thr = threading.Timer(0, self.output)
        self.op_thr.start()

        # listen for commands.
        # only quit command implemented for now
        while self.is_running:
            c = input()
            if (c == 'q'):
                self.stop()

    def calc(self):
        calc_period = max(1,int(self.period/self.speed))
        self.calc_thr = threading.Timer(calc_period, self.calc)
        self.calc_thr.start()
        self.t += self.period
        self.r, self.v = propagate(self.kep,self.epoch,self.t)

    def output(self):
        self.op_thr = threading.Timer(self.op_period, self.output)
        self.op_thr.start()
        self.op_func(self.t,self.r,self.v)

    def _print(_,t,r,v):   # default output method
        print(t,r,v)
    
    def stop(self):
        if self.calc_thr is not None:
            self.calc_thr.cancel()
        if self.op_thr is not None:
            self.op_thr.cancel()
        self.is_running = False

def sig_handler(simulator, signal, frame):
    simulator.stop()
    sys.exit(0)

def print_r(t,r,v):
    print(t,*r)

def print_lat_lon(t,r,v):
    print(t,conv_to_ecef(np.array([[t,r[0],r[1],r[2]]])))

def print_ra_dec(t,r,v):
    print(t,np.degrees(np.arctan2(r[1],r[0])),
            np.degrees(np.arctan2(r[2],(r[0]**2+r[1]**2)**0.5)))

if __name__ == "__main__":
    epoch = 1529410874
    iss_kep = np.array([6775,0.0002893,51.6382,211.1340,7.1114,148.9642])
    s = Simulator(iss_kep,epoch,1,t0=1529433741,speed=1,op_func=print_lat_lon)
    signal.signal(signal.SIGINT, partial(sig_handler,s))
    s.simulate()
