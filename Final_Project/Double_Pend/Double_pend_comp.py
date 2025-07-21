import os
import numpy as np
import casadi as cs
from casadi import Function, MX
import torch
from l4casadi import L4CasADi as NNC
from orc.Final_Project.OCP_solve import OCP_Terminal_Cost, OCP_cost
from orc.Final_Project.Double_Pend.Double_pend_training import Pendulum,ExtendedNeuralNetwork
from example_robot_data.robots_loader import load
import matplotlib.pyplot as plt
import math
import multiprocessing

# Load the double pendulum robot model
robot = load("double_pendulum")

# Model parameters
nq = robot.model.nq  # Number of joints
nv = robot.model.nv  # Number of velocities
nu = 2               # Number of controls (torques)
nx = 2 * nq          # Number of states

# Simulation parameters
dt = 0.01                      # Time step (for optimal control problem)
N = 200                        # Long horizon
M = math.ceil(0.35*N)          # Shorter horizon
K = 50                         # Number of initial states for comparison

# Cost weights
w_p = 1e-1
w_v = 1e-2
w_a = 1e-2
w_term_cost = 1.0  # Terminal cost weight
JMAX = 0  # J_max initialization

# set parallel workers 

n_cores = multiprocessing.cpu_count() - 4 

# Pendulum parameters and dynamics initialization


params_pend = {
    'nq': nq,
    'L1': robot.model.jointPlacements[1].translation[2],
    'L2': robot.model.jointPlacements[2].translation[2],
    'm1': robot.model.inertias[1].mass,
    'm2': robot.model.inertias[2].mass,
    'g' : 9.81,
    'J1': robot.model.inertias[1].inertia[0,0],
    'J2': robot.model.inertias[2].inertia[0,0],
    'b1': 0.05,
    'b2': 0.05,
    'COM1': robot.model.inertias[1].lever[2],
    'COM2': robot.model.inertias[2].lever[2]
}

d_pend = Pendulum(params_pend)

f = d_pend.dynamics(cs.MX.sym('q', nq), cs.MX.sym('dq', nq), cs.MX.sym('u', nu))

# Load the trained neural network model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network model
HiddenLayers = 128
OutputLayers = 1
activation = torch.nn.Tanh()
ModelNN = ExtendedNeuralNetwork(nx, HiddenLayers, OutputLayers, activation, ub=1)
checkpoint = torch.load("/home/student/shared/orc/Final_Project/Double_Pend/nn_cost_pred.pt", map_location=device)
ModelNN.load_state_dict(checkpoint["model_state_dict"])
ModelNN.eval()  # Set to evaluation mode
ModelNN.to(device)

# Convert neural network to CasADi function
build_path = os.path.expanduser("/home/student/shared/orc/Final_Project/Double_Pend/_l4c_gen")
os.makedirs(build_path, exist_ok=True)

l4c_model = NNC(ModelNN, device=str(device), build_dir=build_path)
x_sym = MX.sym("x", 1, nx)  # Symbolic input for CasADi 
nn_terminal_cost = Function("nn_terminal_cost", [x_sym], [JMAX * (l4c_model(x_sym))])

# Arrays to store results
differences = []
x0_vals = []
cmsM = []
cmsMJ = []
cmsNM = []
# Initialize lists to store costs
JM_vec = []
JNM_vec = []


# Loop over K initial states
for _ in range(K):
    x0_val = np.concatenate([
        np.random.uniform(low=-np.pi, high=np.pi, size=nq), 
        np.random.uniform(low=-2.*np.pi, high=2.*np.pi, size=nq)
    ])  # Random initial state
    x0_vals.append(x0_val)

    
    J_optM = OCP_cost(M, x0_val[:nq], x0_val[nq:], nx, nq, w_p, w_v, w_a, dt, f)
    J_optNM = OCP_cost(M + N, x0_val[:nq], x0_val[nq:], nx, nq, w_p, w_v, w_a, dt, f)

    JM_vec.append(J_optM)
    JNM_vec.append(J_optNM)

    # Calculate Medium Cost for each step
    cmsMs = J_optM/ M 
    cmsM.append(cmsMs)
    cmsNMs = J_optNM/ (N+M)
    cmsNM.append(cmsNMs)

JMAX = np.max(cmsNM) # Update JMAX based on computed costs

for _ in range(K):
    
    x0_val = np.array(x0_vals[_])  # Use the same initial state for the terminal cost OCP
    # Define the OCP for the shorter horizon M using terminal cost OCP_Terminal_Cost(N,q0,dq0,nx,nu,w_p,w_v,w_a,dt,f,w_term,terminal_cost_func)
    ocp_terminal_cost = OCP_Terminal_Cost(
        N, 
        q0=x0_val[:nq], 
        dq0=x0_val[nq:], 
        terminal_cost_func=nn_terminal_cost, 
        nx=nx, 
        nu=nu, 
        w_p=w_p, 
        w_v=w_v, 
        w_term=w_term_cost, 
        w_a=w_a, 
        dt=dt, 
        f=f
    )
    cmsMJ.append(ocp_terminal_cost/ (N+M))  # Average cost per step for the shorter horizon

# Calculate the differences between terminal cost and optimal cost
for i in range(K):
    diff = cmsMJ[i] - cmsNM[i]/ cmsNM[i]  # |Terminal Cost - Optimal Cost| / Optimal Cost
    differences.append(diff)
 
# Plot differences
plt.figure()
plt.plot(range(1, K+1), differences, marker='o', linestyle='-', label='(Terminal Cost - Optimal Cost) / Optimal Cost')
plt.xlabel('Initial State Index (K)')
plt.ylabel('Cost Difference')
plt.title('Normalized Cost Difference (M+J) - (N+M)')
plt.legend()
plt.grid()
plt.show()

# Plot cmsM, cmsMJ, and cmsNM
plt.figure()
plt.plot(range(1, K+1), cmsM, marker='o', linestyle='-', label='cmsM (case 1)')
plt.plot(range(1, K+1), cmsMJ, marker='s', linestyle='--', label='cmsMJ (case 2)')
plt.plot(range(1, K+1), cmsNM, marker='^', linestyle='-.', label='cmsNM (case 3)')
plt.xlabel('Initial State Index (K)')
plt.ylabel('Cost Values')
plt.title('Average Cost per Step')
plt.legend()
plt.grid()
plt.show()