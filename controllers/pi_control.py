"""
pid_control
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import numpy as np


class PIControl:
    def __init__(self, kp=0.0, ki=0.0, Ts=0.01, limit=1.0, init_integrator=0.0,lower=None):
        self.kp = kp
        self.ki = ki
        self.Ts = Ts
        self.limit = limit
        self.lower=-limit
        if lower is not None:
            self.lower = lower
        self.integrator = init_integrator
        self.error_delay_1 = 0.0

    def update(self, y_ref, y):

        # compute the error
        error = y_ref - y
        # update the integrator using trapazoidal rule
        self.integrator = self.integrator \
                          + (self.Ts/2) * (error + self.error_delay_1)
        # PI control
        u = self.kp * error \
            + self.ki * self.integrator
        # saturate PI control at limit
        u_sat = self._saturate(u)
        # integral anti-windup
        #   adjust integrator to keep u out of saturation
        if np.abs(self.ki) > 0.0001:
            self.integrator = self.integrator \
                              + (self.Ts / self.ki) * (u_sat - u)
        # update the delayed variables
        self.error_delay_1 = error
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if u >= self.limit:
            u_sat = self.limit
        elif u <= self.lower:
            u_sat = self.lower
        else:
            u_sat = u
        return u_sat
