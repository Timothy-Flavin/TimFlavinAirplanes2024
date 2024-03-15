"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion, quaternion_to_euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


cos = np.cos
sin = np.sin
tan = np.tan

def sec(a):
    return 1/np.cos(a)

def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = quaternion_to_euler(trim_state[6:10])

    ###### TODO ######
    va = mav._Va
    thing = MAV.rho*(va**2)*MAV.S_wing
    # define transfer function constants
    a_phi1 = -1/2*thing*MAV.b*MAV.C_p_p*MAV.b/(2*va)
    a_phi2 = 1/2*thing*MAV.b*MAV.C_p_delta_a
    a_theta1 = -thing*MAV.c/(2*MAV.Jy)*MAV.C_m_q*MAV.c/(2*va)
    a_theta2 = -thing*MAV.c/(2*MAV.Jy)*MAV.C_m_alpha
    a_theta3 = thing*MAV.c/(2*MAV.Jy)*MAV.C_m_delta_e

    # Compute transfer function coefficients using new propulsion model
    a_V1 = MAV.rho*Va_trim*MAV.S_wing/MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha*alpha_trim+MAV.C_D_delta_e*trim_input.elevator) - 1/MAV.mass*dT_dVa(mav,Va_trim,trim_input.throttle)
    a_V2 = 1/MAV.mass*dT_ddelta_t(mav,Va_trim,trim_input.throttle)
    a_V3 = MAV.gravity*cos(theta_trim-alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    
    ##### TODO #####
    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # extract longitudinal states (u, w, q, theta, pd)
    A_lon = A[[3,5,10,7,2],:][:,[3,5,10,7,2]]
    #A_lon[4,:] = -A_lon[4,:]
    #A_lon[:4] = -A_lon[:4]
    B_lon = np.zeros((5,2))
    # change pd to h

    # extract lateral states (v, p, r, phi, psi)
    A_lat = A[[4,9,11,6,8],:][:,[4,9,11,6,8]]
    B_lat = np.zeros((5,2))
    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    
    ##### TODO DONE #####
    x_euler = np.zeros((12,1))
    #print(x_quat)
    #print(x_euler)
    x_euler[6:9,0] = quaternion_to_euler(x_quat[6:10,0])
    x_euler[0:6,0] = x_quat[0:6,0]
    x_euler[9:12,0] = x_quat[10:13,0]
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions

    ##### TODO DONE #####
    x_quat = np.zeros((13,1))
    x_quat[6:10] = euler_to_quaternion(x_euler[6,0],x_euler[7,0],x_euler[8,0])
    x_quat[0:6] = x_euler[0:6]
    x_quat[10:13] = x_euler[9:12]
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state, f_euler will be f, except for the attitude states

    # need to correct attitude states by multiplying f by
    # partial of quaternion_to_euler(quat) with respect to quat
    # compute partial quaternion_to_euler(quat) with respect to quat
    # dEuler/dt = dEuler/dquat * dquat/dt
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    ##### TODO #####
    f_euler_ = np.zeros((12,1))
    
    p_n,p_e,h,u,v,w,phi,theta,psi,p,q,r = x_euler[:,0]
    e0,e1,e2,e3 = x_quat[6:10,0]
    #print(x_euler)
    #print(f"{(p_n,p_e,h,u,v,w,phi,theta,psi,p,q,r)}")
    #print(f"{phi, theta, psi}")
    #input()
    #exit()
    #print(mav._alpha)
    #input()
    p_n_dot =  u*(e0**2 + e1**2 - e2**2 - e3**2) +v*2*(e1*e2-e0*e3) +w*2*(e1*e3+e2*e0)
    p_e_dot  = u*2*(e1*e2+e0*e3) + v*(e0**2-e1**2+e2**2-e3**2) + w*2*(e2*e3-e0*e1)
    p_d_dot  = u*2*(e1*e3-e0*e2) + v*2*(e2*e3+e0*e1) + w*(e0**2-e1**2-e2**2+e3**2)
    h_dot = -p_d_dot
    #p_n_dot = cos(theta)*cos(psi)*u + \
    #          (sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi))*v + \
    #          (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))*w

    #Calculating some C stuff that I need later
    # From ch3 or 4 project:
    m_minus = np.exp(-MAV.M*(mav._alpha-MAV.alpha0))
    m_plus = np.exp(MAV.M*(mav._alpha+MAV.alpha0))
    sigmoid = (1+m_minus+m_plus)/((1+m_minus)*(1+m_plus))
    CL = (1-sigmoid)*(MAV.C_L_0+MAV.C_L_alpha*mav._alpha) + sigmoid*(2*np.sign(mav._alpha)*(np.sin(mav._alpha)**2)*np.cos(mav._alpha))
    CD = MAV.C_D_p + ((MAV.C_L_0+MAV.C_L_alpha * mav._alpha)**2)/(np.pi * MAV.e * MAV.AR)
    
    #print(f"CL: {CL}, CD: {CD}")
    #input()
    # Cs from page 69
    C_X =        -CD*cos(mav._alpha)           +CL*sin(mav._alpha)
    C_X_q =  -MAV.C_D_q*cos(mav._alpha)        +MAV.C_L_q*sin(mav._alpha)
    C_X_de = -MAV.C_D_delta_e*cos(mav._alpha)  +MAV.C_L_delta_e*sin(mav._alpha)
    C_Z =        -CD*sin(mav._alpha)-CL*cos(mav._alpha)
    C_Z_q =  -MAV.C_D_q*sin(mav._alpha)-MAV.C_L_q*cos(mav._alpha)
    C_Z_de = -MAV.C_D_delta_e*sin(mav._alpha) - MAV.C_L_delta_e*cos(mav._alpha)
    thrust_prop, torque_prop = mav._motor_thrust_torque(mav._Va, delta.throttle)


    #q_bar = 0.5*MAV.rho * mav._Va**2
    #F_lift = q_bar * MAV.S_wing*(CL + MAV.C_L_q*(MAV.c/(2*mav._Va))*q + MAV.C_L_delta_e*delta.elevator)
    #F_drag = q_bar * MAV.S_wing*(CD + MAV.C_D_q*(MAV.c/(2*mav._Va))*q + MAV.C_D_delta_e*delta.elevator)

    # compute longitudinal forces in body frame (fx, fz)
    #print(np.array([[np.cos(self._alpha),-np.sin(self._alpha)],[np.sin(self._alpha),np.cos(self._alpha)]]))
    #print(np.array([-F_drag,-F_lift]))
    #fx_fz = np.array([[np.cos(mav._alpha),-np.sin(mav._alpha)],[np.sin(mav._alpha),np.cos(mav._alpha)]])@np.array([[-F_drag],[-F_lift]])


    thing = MAV.rho*(mav._Va**2)*MAV.S_wing/(2*MAV.mass)
    va = mav._Va
    #print(f"{r*v,q*w , -MAV.gravity*sin(theta),         thing,}*({C_X,C_X_q}*{MAV.c}*{q}/{(2*va),C_X_de}*{delta.elevator}){thrust_prop/MAV.mass}")
    #print(r*v-q*w - MAV.gravity*sin(theta)+         thing*(C_X+C_X_q*MAV.c*q/(2*va)+C_X_de*delta.elevator)+thrust_prop/MAV.mass)
    #input() fx_fz[0,0]/MAV.mass# fx_fz[1,0]/MAV.mass#
    u_dot = r*v-q*w - MAV.gravity*sin(theta)+         thing*(C_X+C_X_q*MAV.c*q/(2*va)+C_X_de*delta.elevator)+thrust_prop/MAV.mass
    v_dot = p*w-r*u + MAV.gravity*cos(theta)*sin(phi)+thing*(MAV.C_Y_0 + MAV.C_Y_beta*mav._beta+MAV.C_Y_p*MAV.b*p/(2*va) + MAV.C_Y_r*MAV.b*r/(2*va) + MAV.C_Y_delta_a*delta.aileron+ MAV.C_Y_delta_r*delta.rudder)
    w_dot = q*u-p*v + MAV.gravity*cos(theta)*cos(phi)+thing*(C_Z+C_Z_q*MAV.c*q/(2*va) + C_Z_de*delta.elevator)

    phi_dot   = p+q*sin(phi)*tan(theta)+r*cos(phi)*tan(theta) 
    theta_dot = q*cos(phi)-r*sin(phi)
    psi_dot   = q*sin(phi)*sec(theta)+r*cos(phi)*sec(theta)

    thing2 = MAV.rho*(mav._Va**2)*MAV.S_wing
    p_dot = MAV.gamma1*p*q-MAV.gamma2*q*r + 1/2*thing2*MAV.b*(MAV.C_p_0 + MAV.C_p_beta*mav._beta + MAV.C_p_p*MAV.b*p/(2*va) + MAV.C_p_r*MAV.b*r/(2*va) + MAV.C_p_delta_a*delta.aileron + MAV.C_p_delta_r*delta.rudder) - MAV.gamma3*torque_prop
    q_dot = MAV.gamma5*p*r-MAV.gamma6*(p*p-r*r)+thing2*MAV.c/(2*MAV.Jy)*(MAV.C_m_0 + MAV.C_m_alpha*mav._alpha + MAV.C_m_q*MAV.c*q/(2*va) + MAV.C_m_delta_e*delta.elevator)
    r_dot = MAV.gamma7*p*q-MAV.gamma1*q*r + 1/2*thing2*MAV.b*(MAV.C_r_0 + MAV.C_r_beta*mav._beta + MAV.C_r_p*MAV.b*p/(2*va) + MAV.C_r_r*MAV.b*r/(2*va) + MAV.C_r_delta_a*delta.aileron + MAV.C_r_delta_r*delta.rudder) - MAV.gamma4*torque_prop
    
    f_euler_[:,0] = p_n_dot,p_e_dot,-h_dot,u_dot, v_dot, w_dot, phi_dot,theta_dot, psi_dot, p_dot, q_dot, r_dot
    return f_euler_

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    eps = 0.01  # deviation
    A = np.zeros((12, 12)) # Jacobian of f wrt x
    f_at_x = f_euler(mav, x_euler,delta=delta)
    for i in range (0, 12):
        x_eps = np.copy(x_euler)
        x_eps[i][0] += eps # add eps to i th s ta te
        f_at_x_eps = f_euler(mav,x_eps, delta)
        df_dxi = (f_at_x_eps - f_at_x) / eps
        A[:, i] = df_dxi [: ,0]
    ##### TODO Done #####
    #A = np.zeros((12, 12))  # Jacobian of f wrt x
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    eps = 0.01  # deviation
    B = np.zeros((12, 4))  # Jacobian of f wrt u

    f_at_x = f_euler(mav, x_euler,delta=delta)
    
    delta.elevator+=eps
    df_dxi = (f_euler(mav,x_euler, delta)-f_at_x)/eps
    B[:, 0] = df_dxi [: ,0]
    delta.elevator-=eps

    delta.aileron+=eps
    df_dxi = (f_euler(mav,x_euler, delta)-f_at_x)/eps
    B[:, 1] = df_dxi [: ,0]
    delta.aileron-=eps

    delta.rudder+=eps
    df_dxi = (f_euler(mav,x_euler, delta)-f_at_x)/eps
    B[:, 2] = df_dxi [: ,0]
    delta.rudder-=eps

    delta.throttle+=eps
    df_dxi = (f_euler(mav,x_euler, delta)-f_at_x)/eps
    B[:, 3] = df_dxi [: ,0]
    delta.throttle-=eps
    ##### TODO #####
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.01
    thrust, _ = mav._motor_thrust_torque(Va, delta_t)
    thrust_eps, __ = mav._motor_thrust_torque(Va+eps, delta_t)
    dT_dVa =(thrust_eps-thrust)/eps
    return dT_dVa

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01
    thrust, _ = mav._motor_thrust_torque(Va, delta_t)
    thrust_eps, __ = mav._motor_thrust_torque(Va, delta_t+eps)
    dT_ddelta_t =(thrust_eps-thrust)/eps
    return dT_ddelta_t
