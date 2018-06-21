import sys
import time
import random
import signal
import threading
from functools import partial
import numpy as np

from orbitdeterminator.propagation.sgp4_prop import propagate
from orbitdeterminator.util.teme_to_ecef import conv_to_ecef

class DGSNSimulator():

    def __init__(self,params):

        self.kep       = params.kep
        self.epoch     = params.epoch
        self.t         = params.t0-params.period
        self.period    = params.period
        self.speed     = params.speed
        self.op_writer = params.op_writer

        if params.dgsn_period is not None and params.dgsn_period > 0:
            self.dgsn_omega = np.pi/params.dgsn_period
            self.dgsn_thresh = params.dgsn_thresh
        else:
            self.dgsn_omega = None

        self.r_jit       = params.r_jit
        self.dgsn_period = params.dgsn_period

        self.r = None

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
        interval = random.randint(1,self.period)
        calc_period = max(0,interval/self.speed)
        self.calc_thr = threading.Timer(calc_period, self.calc)
        self.calc_thr.start()
        self.t += interval
        self.r, _ = propagate(self.kep,self.epoch,self.t)
        self.r = list(self.r)

        self.r[0] += random.uniform(-self.r_jit,self.r_jit)
        self.r[1] += random.uniform(-self.r_jit,self.r_jit)
        self.r[2] += random.uniform(-self.r_jit,self.r_jit)

        if self.dgsn_omega is not None:
            prob = abs(np.cos(self.dgsn_omega*self.t))

            if (prob >= self.dgsn_thresh):
                self.op_writer.write(self.t,self.r)
        else:
            self.op_writer.write(self.t,self.r)

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
    def write(t,r):
        print(t,*r)

    def close(self):
        pass

class print_r(OpWriter):
    pass

class print_lat_lon(OpWriter):
    @staticmethod
    def write(t,r):
        t,lat,lon,alt = conv_to_ecef(np.array([[t,*r]]))[0]
        print("{} {} {} {}".format(int(t),lat,lon,alt))

class save_r(OpWriter):
    def __init__(self, name):
        self.file_name = name
        self.iter = 0

    def open(self):
        self.f = open(self.file_name,"a+")
        self.t = None

    def write(self,t,r):
        if not self.t == t:
            self.f.write("{} {} {} {}\r\n".format(t,*r))
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
    op_writer = print_r()

    r_jit = 0
    dgsn_period = None
    dgsn_thresh = 0.5

if __name__ == "__main__":
    epoch = 1529410874
    iss_kep = np.array([6775,0.0002893,51.6382,211.1340,7.1114,148.9642])

    params = SimParams()
    params.kep = iss_kep
    params.epoch = epoch
    params.period = 20
    params.speed = 100

    params.r_jit = 15
    params.dgsn_period = 1350
    params.dgsn_thresh = 0.7

    params.op_writer = print_r()

    s = DGSNSimulator(params)
    signal.signal(signal.SIGINT, partial(sig_handler,s))
    s.simulate()
