"""Simulates signals from the DGSN. This is like simulator.py but
   gives noisy data. Simulated effects are unequal intervals between
   observations, Gaussian noise in observed coordinates, and gaps in
   between observations. The gaps simulate the time when the station
   will be out of range of the satellite.
"""

import sys
import time
import random
import signal
import threading
from functools import partial
import numpy as np

from orbitdeterminator.propagation.cowell import propagate_state
from orbitdeterminator.util.teme_to_ecef import conv_to_ecef
from orbitdeterminator.util.new_tle_kep_state import kep_to_state

class DGSNSimulator():
    """A class for the simulator."""

    def __init__(self,params):
        """Initializes the simulator.

           Args:
               params: A SimParams object containing kep,t0,t,period,speed,
                       op_writer, dgsn_period, and dgsn_thresh. For a
                       description of the parameters, look at the documentation
                       of the SimParams class.

           Returns:
               nothing
        """

        self.s         = kep_to_state(params.kep).flatten()
        self.t0        = params.epoch
        self.t         = params.t0-params.period
        self.period    = params.period
        self.speed     = params.speed
        self.op_writer = params.op_writer

        self.s = propagate_state(self.s,self.t0,self.t)
        self.t0 = self.t

        if params.dgsn_period is not None and params.dgsn_period > 0:
            self.dgsn_omega = np.pi/params.dgsn_period
            self.dgsn_thresh = params.dgsn_thresh
        else:
            self.dgsn_omega = None

        self.r_jit       = params.r_jit
        self.dgsn_period = params.dgsn_period

        self.calc_thr = None
        self.is_running = False

    def simulate(self):
        """Starts the calculation thread and waits for keyboard input.
           Press q or Ctrl-C to quit the simulator cleanly."""

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
        """Calculates the satellite state at current time and
           calls itself after a certain amount of time."""

        interval = random.randint(1,self.period)
        calc_period = max(0,interval/self.speed)
        self.calc_thr = threading.Timer(calc_period, self.calc)
        self.calc_thr.start()
        self.t += interval
        self.s = propagate_state(self.s,self.t0,self.t)
        self.t0 = self.t
        r = self.s[0:3]

        r[0] += random.gauss(0,self.r_jit)
        r[1] += random.gauss(0,self.r_jit)
        r[2] += random.gauss(0,self.r_jit)
        #r[0] += random.uniform(-self.r_jit,self.r_jit)
        #r[1] += random.uniform(-self.r_jit,self.r_jit)
        #r[2] += random.uniform(-self.r_jit,self.r_jit)

        if self.dgsn_omega is not None:
            prob = abs(np.cos(self.dgsn_omega*self.t))

            if (prob >= self.dgsn_thresh):
                self.op_writer.write(self.t,r)
        else:
            self.op_writer.write(self.t0,r)

    def stop(self):
        """Stops the simulator cleanly."""

        if self.calc_thr is not None:
            self.calc_thr.cancel()
        self.is_running = False
        self.op_writer.close()

def sig_handler(simulator, signal, frame):
    """Ctrl-C handler"""

    simulator.stop()
    sys.exit(0)

class OpWriter():
    """Base output writer class. Inherit this class
       and override the methods."""

    def open(self):
        """Anything that has to be executed before
           starting to write output. Runs once.

           Example: Establishing connection to database
        """

        pass

    @staticmethod
    def write(t,r):
        """This method is called everytime the calc thread
           finishes a computation.

           Args:
               t: the current time of simulation
               s: the state vector at t [rx,ry,rz,vx,vy,vz]
        """

        print(t,*r)

    def close(self):
        """Anything that has to be executed after
           finishing writing the output. Runs once.

           Example: Closing connection to a database
        """

        pass

class print_r(OpWriter):
    """Prints the position vector"""
    pass

class print_lat_lon(OpWriter):
    """Prints the latitude and longitude"""
    @staticmethod
    def write(t,r):
        t,lat,lon,alt = conv_to_ecef(np.array([[t,*r]]))[0]
        print("{} {} {} {}".format(int(t),lat,lon,alt))

class save_r(OpWriter):
    """Saves the position vector to a file"""
    def __init__(self, name):
        self.file_name = name
        self.iter = 0

    def open(self):
        #self.f = open(self.file_name,'a+')
        self.t = None

    def write(self,t,r):
        if not self.t == t:
            self.f = open(self.file_name,'a+')
            self.f.write("{} {} {} {}\r\n".format(t,*r))
            self.f.close()
            self.t = t
            print("\rIteration:",self.iter,end=' '*10)
            self.iter+=1

    def close(self):
        #self.f.close()
        pass

class SimParams():
    """SimParams class. This is just a container for all
       the parameters required to start the simulation.

       Params
       ------
       kep(1x6 numpy array): the intial osculating keplerian elements
       epoch(float): the epoch of the above kep
       period(float): maximum time period between observations
       t0(float): starting time of the simulation
       speed(float): speed of the simulation
       op_writer(OpWriter): output handling object

       r_jit(float): std of Gaussian noise applied to observations
       dgsn_period(float): average time period between gaps
       dgsn_thresh(float): used to control the duration of the gap.
                           it is a number between 0 and 1. a higher
                           number means a bigger gap.
    """

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
    epoch = 1531152114
    #epoch = 1530729961
    #epoch = 1529410874
    #iss_kep = np.array([6775,0.0002893,51.6382,211.1340,7.1114,148.9642])

    iss_kep = np.array([6785.6420,0.0003456,51.6418,290.0933,266.6543,212.4306])
    params = SimParams()
    params.kep = iss_kep
    params.epoch = epoch
    params.period = 1
    params.speed = 10

    params.r_jit = 15
    #params.dgsn_period = 1350
    #params.dgsn_thresh = 0.7

    params.op_writer = save_r('ISS_DGSN.csv')

    s = DGSNSimulator(params)
    signal.signal(signal.SIGINT, partial(sig_handler,s))
    s.simulate()
