"""Kalman Filter to smoothen observations. It continuously reads a file
   where observations are being written and updates its estimate based
   on the observations and the cowell model."""

import time
import numpy as np
from functools import partial
from orbitdeterminator.propagation.cowell import rk4

class KalmanFilter():
    """Kalman Filter class wrapper."""

    @staticmethod
    def Jacobian(s,t0,tf):
        """Numerically computes the Jacobian of rk4(s,t0,tf).

           Args:
               s(1x6 numpy array): the state vector at t0 [rx,ry,rz,vx,vy,vz]
               t0(float): the intial time
               tf(float): the final time

           Returns:
               3x3 numpy matrix: the topleft half of the Jacobian of rk4(s,t0,tf)
        """

        f = partial(rk4,t0=t0,tf=tf)
        F = np.empty((3,3))
        a = np.zeros(6)

        h = 0.0005
        a[0] = h
        F[:,0] = (f(s+a) - f(s-a))[0:3]/2/h

        a[0], a[1] = 0, h
        F[:,1] = (f(s+a) - f(s-a))[0:3]/2/h

        a[1], a[2] = 0, h
        F[:,2] = (f(s+a) - f(s-a))[0:3]/2/h

        return F

    def process(self,s,t0,dgsn_file):
        """The main Kalman Filter. Continuously reads an obervations file and
           updates the state estimate.

           Args:
               s(1x6 numpy array): the state vector [rx,ry,rz,vx,vy,vz]
               t0(float): epoch of s
               dgsn_file(string): path to the observations file

           Returns:
               nothing
        """

        self.s = s
        self.t0 = t0
        self.P = np.diag([900,900,900]) # prediction error
        self.Q = np.diag([100,100,100]) # model error
        self.R = np.diag([900,900,900]) # obs error

        f = open(dgsn_file,'r')
        f.seek(0,2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue

            state = line.split()
            if state[0] == '#':
                continue

            t = int(state[0])
            z = [float(x) for x in state[1:4]]

            # predict
            self.s = rk4(self.s, self.t0, t)

            #if (t-self.t0 > 100):
            #    self.s[0:3] = z
            #    self.t0 = t
            #    continue

            F = self.Jacobian(self.s, self.t0, t)

            self.P = np.matmul(self.P,F.T)
            self.P = np.matmul(F,self.P)
            self.P = self.P + self.Q

            # update
            y = z - self.s[0:3]
            S_inv = np.linalg.inv(self.P + self.R)
            K = np.matmul(self.P, S_inv)
            self.s[0:3] = self.s[0:3] + np.matmul(K,y)
            self.P = np.matmul((np.eye(3) - K),self.P)

            self.t0 = t
            print(t,z[0],self.s[0])


if __name__ == '__main__':
    s = np.array([2.87327861e+03,5.22872234e+03,3.23884457e+03,-3.49536799e+00,4.87267295e+00,-4.76846910e+00])
    t0 = 1531152114

    KalmanFilter().process(s,t0,'ISS_DGSN.csv')
