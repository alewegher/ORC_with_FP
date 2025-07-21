"""Configuration file for the double pendulum simulation."""

import numpy as np 
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

T_SIMULATION = 2             # number of time steps simulated
dt = 0.01                    # controller time step
ndt = 10                     # number of dynamics points inside refresh period of controller input 

q0 = np.array([0.0, 0.0])  # initial joint positions

simulate_coulomb_friction = 0
simulation_type = 'timestepping' #either 'timestepping' or 'euler'
tau_coulomb_max = 1.0*np.ones(2) # expressed as percentage of torque max

randomize_robot_model = 0
model_variation = 30.0

use_viewer = True
which_viewer = 'meshcat'
simulate_real_time = True
show_floor = False
PRINT_T = 1                   # print every PRINT_N time steps
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]
