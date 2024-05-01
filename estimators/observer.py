"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import numpy as np
from scipy import stats
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
import parameters.aerosonde_parameters as MAV
import scipy
import control

class Observer:
    def __init__(self, ts_control, initial_measurements = MsgSensors()):
        # initialized estimated state message
        self.estimated_state = MsgState()
        # use alpha filters to low pass filter gyros and accels
        # alpha = Ts/(Ts + tau) where tau is the LPF time constant

        ##### TODO #####
        self.lpf_gyro_x = AlphaFilter(alpha=0.99, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.99, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.99, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_z)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.98, y0=MAV.rho*MAV.gravity*100)
        self.lpf_diff = AlphaFilter(alpha=0.7, y0=initial_measurements.diff_pressure)
        # ekf for phi and theta
        self.attitude_ekf = EkfAttitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = EkfPosition()

    def update(self, measurement, true_state=None):
        ##### TODO #####
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x)#self.lpf_gyro_x(measurex, state)
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z)

        # invert sensor model to get altitude and airspeed
        self.estimated_state.altitude = self.lpf_abs.update(measurement.abs_pressure)/(MAV.rho*MAV.gravity)
        self.estimated_state.Va = np.sqrt(2/MAV.rho*self.lpf_diff.update(measurement.diff_pressure))

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(measurement, self.estimated_state)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(measurement, self.estimated_state)

        # not estimating these
        self.estimated_state.alpha = true_state.alpha
        self.estimated_state.beta = true_state.beta
        self.estimated_state.gamma = true_state.gamma
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state


class AlphaFilter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        ##### TODO #####
        self.y = self.alpha * self.y + (1-self.alpha) * u
        return self.y


class EkfAttitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        ##### TODO #####
        self.Q = np.diag([(SENSOR.gyro_sigma**2) / 5, SENSOR.gyro_sigma**2])
        self.Q_gyro = np.diag([0, 0, 0]) 
        self.R_accel = np.diag([SENSOR.gyro_sigma, SENSOR.gyro_sigma, SENSOR.gyro_sigma])
        self.N = 1  # number of prediction step per sample
        self.xhat = np.array([[0.0], [0.0]]) # initial state: phi, theta
        self.P = np.diag([0, 0])
        self.Ts = SIM.ts_control/self.N
        self.gate_threshold = 0 #stats.chi2.isf(q=?, df=?)

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        ##### TODO #####
        sin = np.sin
        cos = np.cos
        tan = np.tan
        p = state.p
        q = state.q
        r = state.r
        phi = x.item(0)
        theta = x.item(1)
        
        f_ = np.zeros((2,1))
        f_[0,0] = p+q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
        f_[1,0] = q*cos(phi)-r*sin(phi)
        return f_

    def h(self, x, measurement, state):
        # measurement model y
        ##### TODO #####
        sin = np.sin
        cos = np.cos
        p = state.p
        q = state.q
        r = state.r
        phi = x.item(0)
        theta = x.item(1)
        Va = state.Va


        h_ = np.array( [[q*Va*sin(theta)+MAV.gravity*sin(theta)],  # x-accel
                        [r*Va*cos(theta)-p*Va*sin(theta)-MAV.gravity*cos(theta)*sin(phi)],# y-accel
                        [-q*Va*cos(theta)-MAV.gravity*cos(theta)*cos(phi)]])  # z-accel
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        ##### TODO ##### Algorithm 3
        Tout = self.Ts
        for i in range(0, self.N):
            Tp = Tout/self.N
            self.xhat += Tout/self.N*self.f(self.xhat,measurement,state)
            A = jacobian(self.f, self.xhat, measurement, state)
            Ad = np.identity(A.shape[0]) + A*Tp + A@A*Tp*Tp
            self.P = Ad@self.P@Ad.T + Tp*Tp*self.Q

    def measurement_update(self, measurement, state):
        # measurement updates
        h = self.h(self.xhat, measurement, state)
        C = jacobian(self.h, self.xhat, measurement, state)
        y = np.array([[measurement.accel_x, measurement.accel_y, measurement.accel_z]]).T

        ##### TODO ##### from book
        S = self.R_accel+C@self.P @ C.T
        #print(f" r accel: {self.R_accel}, C: {C}, self.P: {self.P} = {S}")
        S_inv = np.linalg.inv(S)

        if stats.chi2.sf((y-h).T@S_inv@(y-h), df=3)>0.01:#(y-h).T @ S_inv @ (y-h) < self.gate_threshold:
            L = self.P@C.T@S_inv 
            tmp = np.eye(2) - L@C
            self.P = tmp@self.P@tmp.T+ L@self.R_accel@L.T
            self.xhat = self.xhat + L @ (y-h)


class EkfPosition:
    # implement continous-discrete EKF to estimate pn, pe, Vg, chi, wn, we, psi
    def __init__(self):
        self.Q = np.diag([
                    0.9,  # pn
                    0.9,  # pe
                    0.5,  # Vg
                    0.1, # chi
                    0.1, # wn
                    0.1, # we
                    0.1, #0.0001, # psi
                    ])
        self.R_gps = np.diag([
                    SENSOR.gps_n_sigma,  # y_gps_n
                    SENSOR.gps_e_sigma,  # y_gps_e
                    SENSOR.gps_Vg_sigma,  # y_gps_Vg
                    SENSOR.gps_course_sigma,  # y_gps_course
                    ])
        self.R_pseudo = np.diag([
                    0.1,  # pseudo measurement #1
                    0.1,  # pseudo measurement #2
                    ])
        self.N = 1  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat = np.array([[0.0], [0.0], [30.0], [0.0], [0.0], [0.0], [0.0]])
        self.P = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.gps_n_old = 0
        self.gps_e_old = 0
        self.gps_Vg_old = 30
        self.gps_course_old = 0
        self.pseudo_threshold = 0 #stats.chi2.isf(q=?, df=?)
        self.gps_threshold = 100000 # don't gate GPS

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.north = self.xhat.item(0)
        state.east = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        cos = np.cos
        sin = np.sin 
        psi = x.item(6)
        psi_dot = state.q*sin(state.phi)/cos(state.theta) + state.r*cos(state.phi)/cos(state.theta)
        Vg = x.item(2)#measurement.gps_Vg
        Chi = x.item(3)#measurement.gps_course
        f_ = np.array([[Vg*cos(Chi)],
                       [Vg*sin(Chi)],
                       [(state.Va*cos(psi) * (-state.Va*psi_dot*sin(psi)) + \
                        state.Va*sin(psi) * (state.Va*psi_dot*cos(psi)))/Vg],
                       [MAV.gravity/Vg*np.tan(state.phi)],
                       [0.0],
                       [0.0],
                       [psi_dot], ###TIM THING
                       ])
        return f_

    def h_gps(self, x, measurement, state):
        # measurement model for gps measurements
        h_ = np.array([
            [x.item(0)], #pn
            [x.item(1)], #pe
            [x.item(2)], #Vg
            [x.item(3)], #chi
        ])
        return h_

    def h_pseudo(self, x, measurement, state):
        # measurement model for wind triangale pseudo measurement
        h_ = np.array([
            [state.Va*np.cos(x.item(6)) - measurement.gps_Vg*np.cos(x.item(3))],  # wind triangle x
            [state.Va*np.sin(x.item(6)) - measurement.gps_Vg*np.sin(x.item(3))],  # wind triangle y
        ])
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        self.Ts
        for i in range(0, self.N):
            # propagate model
            Tp = self.Ts
            self.xhat = self.xhat + self.Ts * self.f(self.xhat,measurement,state)
            # compute Jacobian
            A = jacobian(self.f,self.xhat,measurement,state)
            # convert to discrete time models
            Ad = np.identity(A.shape[0]) + A*Tp + A@A*Tp*Tp
            self.P = Ad@self.P@Ad.T + Tp*Tp*self.Q

    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, measurement, state)
        C = jacobian(self.h_pseudo, self.xhat, measurement, state)
        y = np.array([[0, 0]]).T
        S_inv = np.linalg.inv(self.R_pseudo + C@self.P@C.T)
        #print(S_inv)
        #print(C@self.P@C.T)
        if (y-h).T @ S_inv @ (y-h) < self.pseudo_threshold:
            Li = self.P@C.T@S_inv
            #print(Li)
            #input()
            blah = np.identity(2)-Li@C
            self.P = blah@self.P@blah.T + Li@self.R_gps@Li.T
            self.xhat = self.xhat + Li@(y-h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            h = self.h_gps(self.xhat, measurement, state)
            C = jacobian(self.h_gps, self.xhat, measurement, state)
            y_chi = wrap(measurement.gps_course, h[3, 0])
            y = np.array([[measurement.gps_n,
                           measurement.gps_e,
                           measurement.gps_Vg,
                           y_chi]]).T
            S_inv = np.linalg.inv(self.R_gps + C@self.P@C.T)#np.zeros((4,4))
            #print(S_inv)
            #print((y-h).T @ S_inv @ (y-h))
            if (y-h).T @ S_inv @ (y-h) < self.gps_threshold*100:
                #print("Updating xhat")
                Li = self.P@C.T@S_inv
                #print(f"{Li} = {self.P}, {C}, {S_inv}")
                blah = np.identity(7)-Li@C
                self.P = blah@self.P@blah.T + Li@self.R_gps@Li.T
                self.xhat = self.xhat + Li@(y-h)
                

            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course


def jacobian(fun, x, measurement, state):
    # compute jacobian of fun with respect to x
    f = fun(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.0001  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, measurement, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J