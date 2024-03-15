"""
mavsimPy
    - Chapter 4 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/27/2018 - RWB
        1/17/2019 - RWB
        1/5/2023 - David L. Christiansen
        7/13/2023 - RWB
"""
import os, sys
# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0,os.fspath(Path(__file__).parents[2]))
# use QuitListener for Linux or PC <- doesn't work on Mac
#from python_tools.quit_listener import QuitListener
import pyqtgraph as pg
import parameters.simulation_parameters as SIM
from models.compute_models import compute_ss_model
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from viewers.mav_viewer import MavViewer
from viewers.data_viewer import DataViewer
from message_types.msg_delta import MsgDelta
import numpy as np
import matplotlib.pyplot as plt

import pygame
#quitter = QuitListener()
pygame.init()
VIDEO = False
PLOTS = True
ANIMATION = True
SAVE_PLOT_IMAGE = False

if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap4_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

#initialize the visualization
if ANIMATION or PLOTS:
    app = pg.QtWidgets.QApplication([]) # use the same main process for Qt applications
if ANIMATION:
    mav_view = MavViewer(app=app)  # initialize the mav viewer
if PLOTS:
    # initialize view of data plots
    data_view = DataViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation)
delta = MsgDelta()

# create init parameters
Va0 = 30
alpha0 = 0.
beta0 = 0.
mav.initialize_velocity(Va0,alpha0,beta0)

# initialize the simulation time
sim_time = SIM.start_time
plot_time = sim_time
end_time = 60

from timstuff import trim
alpha, elevator, throttle = trim.compute_trim(mav,delta)
mav.initialize_velocity(Va0,alpha,beta0)
# main simulation loop
display = pygame.display.set_mode((300, 300))
print("Press 'Esc' to exit...")
keys = {"a":0,'d':0,'w':0,'s':0,'q':0,'e':0,'x':0,'z':0}

print(mav._state.shape)
trim_state = mav._state[:13]
print(trim_state.shape)

trim_input = MsgDelta(elevator=elevator,
                        aileron=0,
                        rudder=0,
                        throttle=throttle)
alon, blon, alat, blat = compute_ss_model(mav,trim_state,trim_input)
print(f"{alon,blon,alat,blat}")
eigs = np.linalg.eigvals(alon)
print(eigs)
# extract real part 
x = [ele.real for ele in eigs] 
# extract imaginary part 
y = [ele.imag for ele in eigs] 

for xi,yi in zip(x,y):
    w = np.sqrt(xi**2+yi**2)
    print(f"w: {w} and damping: {-xi/w}")

#2 Reigs + abs|eig|^2 =2 damp w s + w^2 
    
# damp = R(eig) / w
    
# a = 1, b = 2damp*s, c = s^2

#-2damp*s +- sqrt(4*damp^2*s^2 - 4*1*s^2)
# -------------------------------------
#                  2
    
#-2damp*s +- sqrt(4*s^2(damp^2-1))
# -------------------------------------
#                  2
plt.scatter(x,y)
plt.ylabel('Imaginary') 
plt.xlabel('Real')
#plt.show()
input()

while sim_time < end_time:
    
    # ------- set control surfaces -------------
    delta.elevator = elevator#-0.1248
    if(abs(sim_time-3.0) < 0.1):
        delta.elevator+=0.5 # excite the modes
    delta.aileron = 0# 0.001836
    delta.rudder = 0# -0.0003026
    delta.throttle = throttle# 0.6768

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
        # checking if keydown event happened or not
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                keys['d']=1
            if event.key == pygame.K_a:
                keys['a']=1
            if event.key == pygame.K_w:
                keys['w']=1
            if event.key == pygame.K_s:
                keys['s']=1
            if event.key == pygame.K_q:
                keys['q']=1
            if event.key == pygame.K_e:
                keys['e']=1
            if event.key == pygame.K_z:
                keys['z']=1
            if event.key == pygame.K_x:
                keys['x']=1
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_d:
                keys['d']=0
            if event.key == pygame.K_a:
                keys['a']=0
            if event.key == pygame.K_w:
                keys['w']=0
            if event.key == pygame.K_s:
                keys['s']=0
            if event.key == pygame.K_q:
                keys['q']=0
            if event.key == pygame.K_e:
                keys['e']=0
            if event.key == pygame.K_z:
                keys['z']=0
            if event.key == pygame.K_x:
                keys['x']=0
    
        delta.elevator += 0.3*(int(keys['w'])-int(keys['s']))
        delta.aileron += 0.5*(int(keys['e'])-int(keys['q']))
        delta.rudder += 0.3*(int(keys['d'])-int(keys['a']))
        delta.throttle += 0.32*(int(keys['z'])-int(keys['x']))

    # ------- physical system -------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # -------update viewer-------------
    if ANIMATION:
        mav_view.update(mav.true_state)  # plot body of MAV
    if PLOTS:
        plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                            None,  # estimated states
                            None,  # commanded states
                            delta)  # inputs to aircraft
    if ANIMATION or PLOTS:
        app.processEvents()
    if VIDEO is True:
        video.update(sim_time)
        
    # # -------Check to Quit the Loop-------
    # if quitter.check_quit():
    #     break

    # -------increment time-------------
    sim_time += SIM.ts_simulation


if SAVE_PLOT_IMAGE:
    data_view.save_plot_image("ch4_plot")

if VIDEO is True:
    video.close()