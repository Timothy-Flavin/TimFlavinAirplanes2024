"""
mavsimPy
    - Chapter 2 launch file for Beard & McLain, PUP, 2012
    - Update history:  
        12/27/2018 - RWB
        1/17/2019 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
        1/16/2024 - RWB
"""
import os, sys
import numpy as np
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from python_tools.quit_listener import QuitListener
import pyqtgraph as pg
import parameters.simulation_parameters as SIM
from viewers.mav_viewer import MavViewer
from message_types.msg_state import MsgState

#quitter = QuitListener()
VIDEO = False
if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap2_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)
# initialize the visualization
app = pg.QtWidgets.QApplication([])
mav_view = MavViewer(app=app)  # initialize the mav viewer
# initialize elements of the architecture
state = MsgState()

# initialize the simulation time
sim_time = SIM.start_time
motions_time = 0
time_per_motion = 1
end_time = 20

# main simulation loop
print("Press Esc to exit...")

def V_to_B(psi, theta, phi):
    return np.array([[np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(psi)],
                     [np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.sin(phi)*np.cos(theta)],
                     [np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(theta)],])

def B_to_V(psi, theta, phi):
    R_b_v2 = np.array([[1,0,0],
                       [0,np.cos(phi),-np.sin(phi)],
                       [0,np.sin(phi),np.cos(phi)],])
    R_v2_v1 = np.array([[np.cos(theta),0,np.sin(theta)],
                        [0,1,0],
                        [-np.sin(theta),0,np.cos(theta)]])
    R_v1_v = np.array([[np.cos(psi),-np.sin(psi),0],
                        [np.sin(psi),np.cos(psi),0],
                        [0,0,1]])
    
    return R_v1_v@R_v2_v1@R_b_v2

pt = np.array([[1],[0],[0]])
print(pt)
print(V_to_B(np.deg2rad(90),0,0)@pt) # should be 0,-1,0
print(V_to_B(0,np.deg2rad(90),0)@pt) # should be 0,0,1
print(V_to_B(0,0,np.deg2rad(90))@pt) # should be 1,0,0
print()

print(B_to_V(np.deg2rad(90),0,0)@V_to_B(np.deg2rad(90),0,0)@pt) # should be 0,1,0
print(B_to_V(0,np.deg2rad(90),0)@V_to_B(0,np.deg2rad(90),0)@pt) # should be 0,0,1
print(B_to_V(0,0,np.deg2rad(90))@V_to_B(0,0,np.deg2rad(90))@pt) # should be 1,0,0

x_pt = np.array([[1],[0],[0]])
y_pt = np.array([[0],[1],[0]])
z_pt = np.array([[0],[0],[1]])
while sim_time < end_time:
    # -------vary states to check viewer-------------
    if motions_time < time_per_motion:
        state.north += 10*SIM.ts_simulation
    elif motions_time < time_per_motion*2:
        state.east += 10*SIM.ts_simulation
    elif motions_time < time_per_motion*3:
        state.altitude += 10*SIM.ts_simulation
    elif motions_time < time_per_motion*4:
        state.psi += 0.1*SIM.ts_simulation
    elif motions_time < time_per_motion*5:
        state.theta += 0.1*SIM.ts_simulation
    elif motions_time < time_per_motion*6:
        state.phi += 0.1*SIM.ts_simulation
    elif motions_time < time_per_motion*10:
        state.psi+=0*SIM.ts_simulation
        state.theta+=0.5*SIM.ts_simulation
        state.phi+=-0.025*SIM.ts_simulation
        forward_move = B_to_V(state.psi,state.theta,state.phi)@pt #pt is 1,0,0
        state.north += 20*forward_move[0,0]*SIM.ts_simulation
        state.east += 20*forward_move[1,0]*SIM.ts_simulation
        state.altitude -= 20*forward_move[2,0]*SIM.ts_simulation

    elif motions_time < time_per_motion*14:
        state.psi+=0.0*SIM.ts_simulation
        state.theta+=0.5*SIM.ts_simulation
        state.phi+=0.0*SIM.ts_simulation
        forward_move = 1.2*B_to_V(state.psi,state.theta,state.phi)@pt #pt is 1,0,0
        state.north += 20*forward_move[0,0]*SIM.ts_simulation
        state.east += 20*forward_move[1,0]*SIM.ts_simulation
        state.altitude -= 20*forward_move[2,0]*SIM.ts_simulation
    
    elif motions_time < time_per_motion*16:
        state.psi+=0.0*SIM.ts_simulation
        state.theta+=0.0*SIM.ts_simulation
        state.phi+=1.5*SIM.ts_simulation
        forward_move = 2.0*B_to_V(state.psi,state.theta,state.phi)@pt #pt is 1,0,0
        state.north += 20*forward_move[0,0]*SIM.ts_simulation
        state.east += 20*forward_move[1,0]*SIM.ts_simulation
        state.altitude -= 20*forward_move[2,0]*SIM.ts_simulation
    
    else:
        state.psi-=0.0*SIM.ts_simulation
        state.theta-=0.25*SIM.ts_simulation
        state.phi+=0.0*SIM.ts_simulation
        forward_move = 2.0*B_to_V(state.psi,state.theta,state.phi)@pt #pt is 1,0,0
        state.north += 20*forward_move[0,0]*SIM.ts_simulation
        state.east += 20*forward_move[1,0]*SIM.ts_simulation
        state.altitude -= 20*forward_move[2,0]*SIM.ts_simulation
    # -------update viewer and video-------------
    mav_view.update(state)
    
    mav_view.process_app()

    # -------increment time-------------
    sim_time += SIM.ts_simulation
    motions_time += SIM.ts_simulation
    #if motions_time >= time_per_motion*6:
        #motions_time = 0

    # -------update video---------------
    if VIDEO is True:
        video.update(sim_time)
    
    # # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

if VIDEO is True:
    video.update(sim_time)