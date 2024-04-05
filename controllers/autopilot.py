"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import numpy as np
from models.mav_dynamics_control import MavDynamics
import parameters.control_parameters as AP
# from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pid_control import PIDControl
from controllers.pd_control import PDControl
from controllers.pd_control_with_rate import PDControlWithRate
from controllers.tf_control import TFControl
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta

airspeed_throttle_kp=0.05
airspeed_throttle_ki=0.05

alpha_elevator_kp = -20.0
alpha_elevator_ki = -20.
alpha_elevator_kd = -1.

yaw_damper_kp = 10.0
yaw_damper_kd = 1.0

psi_aileron_kp = 0.3
psi_aileron_ki = 0.05
psi_aileron_kd = psi_aileron_kp/1

gamma_alpha_kp = 1.2
gamma_alpha_ki = 0.1
gamma_alpha_kd = 0#.0001

alt_gamma_kp = 0.01
alt_gamma_ki = 0.001
alt_gamma_kd = 0.001#alt_gamma_kp/10

chi_psi_kp = 1.0
chi_psi_ki = 0.0001
chi_psi_kd = chi_psi_kp/10
class Autopilot:
    def __init__(self, delta, mav:MavDynamics, ts_control):

        self.throttle_from_airspeed = PIControl(
            kp=airspeed_throttle_kp,
            ki=airspeed_throttle_ki,
            Ts=ts_control,
            init_integrator=delta.throttle/airspeed_throttle_ki,
            limit=1.0,
            lower=0.0)
        self.elevator_from_alpha = PIDControl(
            kp=alpha_elevator_kp,
            ki=alpha_elevator_ki,
            kd=alpha_elevator_kd,
            limit=1.0,
            Ts=ts_control,
            init_integrator=delta.elevator/alpha_elevator_ki,
            )
        self.alpha_from_gamma = PIDControl(
            kp=gamma_alpha_kp,
            ki=gamma_alpha_ki,
            kd=gamma_alpha_kd,
            limit=np.radians(5),
            Ts=ts_control,
            init_integrator=mav._alpha/gamma_alpha_ki,
            )
        self.gamma_from_altitude = PIDControl(
            kp=alt_gamma_kp,
            ki=alt_gamma_ki,
            kd=alt_gamma_kd,
            limit=np.radians(20),
            Ts=ts_control,
            init_integrator=0,
            )

        self.yaw_damper = PDControl(kp=yaw_damper_kp,
                                    kd=yaw_damper_kd,
                                    limit=1,
                                    Ts=ts_control)
        
        self.aileron_from_psi = PIDControl(
            kp=psi_aileron_kp,
            ki=psi_aileron_ki,
            kd=psi_aileron_kd,
            limit=1.0,
            Ts=ts_control,
            init_integrator=delta.aileron/psi_aileron_ki,
            )
        self.psi_from_chi = PIDControl(
            kp=chi_psi_kp,
            ki=chi_psi_ki,
            kd=chi_psi_kd,
            limit=np.radians(45),
            Ts=ts_control,
            init_integrator=0,
            )
        self.h0 = 100
        # instantiate lateral-directional controllers
        # self.roll_from_aileron = PDControlWithRate(
        #                 kp=AP.roll_kp,
        #                 kd=AP.roll_kd,
        #                 limit=np.radians(45))
        # self.course_from_roll = PIControl(
        #                 kp=AP.course_kp,
        #                 ki=AP.course_ki,
        #                 Ts=ts_control,
        #                 limit=np.radians(30))
        # # self.yaw_damper = TransferFunction(
        # #                 num=np.array([[AP.yaw_damper_kr, 0]]),
        # #                 den=np.array([[1, AP.yaw_damper_p_wo]]),
        # #                 Ts=ts_control)
        # self.yaw_damper = TFControl(
        #                 k=AP.yaw_damper_kr,
        #                 n0=0.0,
        #                 n1=1.0,
        #                 d0=AP.yaw_damper_p_wo,
        #                 d1=1,
        #                 Ts=ts_control)

        # # instantiate longitudinal controllers
        # self.pitch_from_elevator = PDControlWithRate(
        #                 kp=AP.pitch_kp,
        #                 kd=AP.pitch_kd,
        #                 limit=np.radians(45))
        # self.altitude_from_pitch = PIControl(
        #                 kp=AP.altitude_kp,
        #                 ki=AP.altitude_ki,
        #                 Ts=ts_control,
        #                 limit=np.radians(30))
        # self.airspeed_from_throttle = PIControl(
        #                 kp=AP.airspeed_throttle_kp,
        #                 ki=AP.airspeed_throttle_ki,
        #                 Ts=ts_control,
        #                 limit=1.0)
        self.commanded_state = MsgState()

    def update(self, cmd, state:MsgState):
        #print(cmd.airspeed_command)
        #print(state.alpha)
        #input()
        delta = MsgDelta(elevator=0,aileron=0,rudder=0,throttle=0)
        delta.throttle=self.throttle_from_airspeed.update(cmd.airspeed_command,state.Va)

        gamma = self.gamma_from_altitude.update(cmd.altitude_command,state.altitude)
        alpha = self.alpha_from_gamma.update(gamma,state.gamma)
        
        delta.elevator = self.elevator_from_alpha.update(alpha, state.alpha)
        
        #input()
        self.h0=state.altitude
        
	    #### TODO #####
        # lateral autopilot

        delta.rudder = self.yaw_damper.update(0,state.beta)
        # longitudinal autopilot
        #print(cmd.course_command)
        course = state.chi
        if state.chi<0:
            course+=2*np.pi
        #print(course)
        phi = self.psi_from_chi.update(np.radians(cmd.course_command),course)
        delta.aileron = self.aileron_from_psi.update(phi,state.phi)
        # construct control outputs and commanded states
        #print(f"state gamma: {-state.gamma}, command gamma: {gamma}\nAltitude: {state.altitude}, \nalpha command: {alpha},\nelevator: {delta.elevator}\
        #      \ntheta: {state.theta}, delta_h: {state.altitude-self.h0}\n\
        #      chi: {state.chi} chi_command: {np.radians(90)}, \n\
        #      psi_command: {phi}, {state.phi}")
        #input()
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.gamma = gamma
        self.commanded_state.alpha = alpha
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi
        self.commanded_state.theta = 0
        self.commanded_state.chi = np.radians(cmd.course_command)
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
