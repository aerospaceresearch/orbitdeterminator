"""Outputs a satellite's location periodically."""

import sys
import time
import signal
import threading
from functools import partial
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from propagation.cowell import propagate_state
from util.teme_to_ecef import conv_to_ecef
from util.new_tle_kep_state import kep_to_state

class Simulator():
    """A class for the simulator."""

    def __init__(self,params):
        """Initializes the simulator.

           Args:
               params: A SimParams object containing kep,t0,t,period,speed,
                       and op_writer

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

        calc_period = max(0,self.period/self.speed)
        self.calc_thr = threading.Timer(calc_period, self.calc)
        self.calc_thr.start()
        self.t += self.period
        self.s = propagate_state(self.s,self.t0,self.t)
        self.t0 = self.t
        self.op_writer.write(self.t,self.s)

    def stop(self):
        """Stops the simulator cleanly."""

        if self.calc_thr is not None:
            self.calc_thr.cancel()
        self.is_running = False
        self.op_writer.close()

def __sig_handler(simulator, signal, frame):
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
    def write(t,s):
        """This method is called everytime the calc thread
           finishes a computation.

           Args:
               t: the current time of simulation
               s: the state vector at t [rx,ry,rz,vx,vy,vz]
        """

        print(t,*s)

    def close(self):
        """Anything that has to be executed after
           finishing writing the output. Runs once.

           Example: Closing connection to a database
        """

        pass

# Sample op_writer classes
class print_r(OpWriter):
    """Prints the position vector"""
    @staticmethod
    def write(t,s):
        print(t,*s[0:3])

class print_lat_lon(OpWriter):
    """Prints the latitude and longitude"""
    @staticmethod
    def write(t,s):
        t,lat,lon,alt = conv_to_ecef(np.array([[t,*s[0:3]]]))[0]
        print(t,lat,lon,alt)

class save_r(OpWriter):
    """Saves the position vector to a file"""
    def __init__(self, name):
        """Initialize the class.

           Args:
               name(string): file name
        """

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
            self.f.write("{} {} {} {}\r\n".format(t,*s))
            self.f.close()
            self.t = t
            print("\rIteration:",self.iter,end=' '*10)
            self.iter+=1

    def close(self):
        self.f.close()

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

    """

    kep = None
    epoch = None
    period = 1
    t0 = int(time.time())
    speed = 1
    op_writer = OpWriter()

if __name__ == "__main__":
    epoch = 1531152114
    iss_kep = np.array([6785.68682,0.0003456,51.6418,290.0933,266.6543,212.430557])

    params = SimParams()
    params.kep = iss_kep
    params.epoch = epoch
    params.op_writer = print_lat_lon() #save_r('ISS.csv')

    s = Simulator(params)
    signal.signal(signal.SIGINT, partial(__sig_handler,s))
    s.simulate()
