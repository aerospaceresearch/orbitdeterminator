import sys
import time
import signal
import threading
from functools import partial
import numpy as np

from orbitdeterminator.propagation.sgp4_prop import propagate
from orbitdeterminator.util.teme_to_ecef import conv_to_ecef

class Simulator():

    def __init__(self,params):

        self.kep       = params.kep
        self.epoch     = params.epoch
        self.t         = params.t0-params.period
        self.period    = params.period
        self.speed     = params.speed
        self.op_writer = params.op_writer

        self.r = None
        self.v = None

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
        self.r, self.v = propagate(self.kep,self.epoch,self.t)
        self.op_writer.write(self.t,self.r,self.v)

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
    def write(t,r,v):
        print(t,*r,*v)

    def close(self):
        pass

class print_r(OpWriter):
    @staticmethod
    def write(t,r,v):
        print(t,*r)

class print_lat_lon(OpWriter):
    @staticmethod
    def write(t,r,v):
        print(conv_to_ecef(np.array([[t,*r]])))

class save_r(OpWriter):
    def __init__(self, name):
        self.file_name = name

    def open(self):
        self.f = open(self.file_name,"a+")
        self.t = None

    def write(self,t,r,v):
        if not self.t == t:
            self.f.write("{} {} {} {}\r\n".format(t,*r))
            self.t = t

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
    epoch = 1529410874
    iss_kep = np.array([6775,0.0002893,51.6382,211.1340,7.1114,148.9642])

    params = SimParams()
    params.kep = iss_kep
    params.epoch = epoch
    params.op_writer = print_lat_lon()

    s = Simulator(params)
    signal.signal(signal.SIGINT, partial(sig_handler,s))
    s.simulate()
