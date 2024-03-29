import numpy as np
import scipy as sp
from models.mav_dynamics_control import MavDynamics
from message_types.msg_delta import MsgDelta
from scipy.optimize import minimize

def compute_trim(mav=MavDynamics, delta=MsgDelta):

  x0 = [mav._alpha, delta.elevator, delta.throttle]
  bounds = [(0,np.deg2rad(12)),(-1,1),(0,1)]
  res = minimize(mav.calculate_trim_output,x0,bounds=bounds,method='SLSQP')
  #print(res)
  return (res.x[0],res.x[1],res.x[2])
  # parameters to input for the trim 
  # alpha elevator throttle 
  # delta_a = 0.01
  # forces_moments = mav._forces_moments(delta=delta)
  # fx = forces_moments[0]
  # fz = forces_moments[2]
  # m = forces_moments[4]

  # print(f"fx: {fx}, fz: {fz}, m: {m}")
  # mav.initialize_velocity(mav._Va,mav._alpha+delta_a,mav._beta)

  # forces_moments = mav._forces_moments(delta=delta)
  # fx_n = forces_moments[0]
  # fz_n = forces_moments[2]
  # m_n = forces_moments[4]

  # print(f"after fx: {fx_n}, fz: {fz_n}, m: {m_n}")
  # fx_delta = (fx_n-fx)/delta_a
  # fz_delta = (fz_n-fz)/delta_a
  # m_delta = (m_n-m)/delta_a
  # print(f"J fx: {fx_delta}, fz: {fz_delta}, m: {m_delta}")
  # delta_a=-fx/fx_delta
  # compute_parameters(mav,delta,delta_a)
  # input("yoo")

def compute_parameters(mav:MavDynamics,delta:MsgDelta,delta_a:float):
  mav.initialize_velocity(mav._Va, mav._alpha +delta_a, mav._beta)
  forces_moments=mav._forces_moments(delta=delta)
  fx = forces_moments[0]
  fz = forces_moments[2]
  m = forces_moments[4]
  print(f"fx: {fx}, fz: {fz}, m: {m}")
  

def do_trim(mav:MavDynamics,Va,alpha):
  delta=MsgDelta()
  Va0 = Va
  alpha0 = alpha
  beta0 = 0.0
  mav.initialize_velocity(Va0,alpha0, beta0)
  
  #initialize simulation time
  delta.elevator = -0.1248
  delta.aileron=0.0
  delta.rudder=0.0
  delta.throttle = 0.6768

  alpha,elevator,throttle = compute_trim(mav,delta)
  mav.initialize_velocity(Va0,alpha,beta0)
  delta.elevator = elevator
  delta.throttle = throttle
  return delta