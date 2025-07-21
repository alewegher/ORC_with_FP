import os
import numpy as np
import casadi as cs
from casadi import Function, MX
import torch
from orc.Final_Project.neural_network import NeuralNetwork
from l4casadi import L4CasADi as NNC
from orc.Final_Project.OCP_solve import OCP_Terminal_Cost,OCP_cost
from orc.Final_Project.Single_Pend.SinPendTrain import SimPend
import matplotlib.pyplot as plt
import math 

# parameters definition for the pendulum dynamics

g = 9.81                # gravity constant
L = 0.8                 # length of the pendulum
m = 0.25                # mass of the pendulum
b = 0.1                 # damping coefficient
Jp = 1/3* m * L**2      # moment of inertia of the pendulum --> assuming the pendulum is a rod of length L and mass m, pivoted at one end --> COM is at L/2

nj = 1       # number of joints
nx = 2 * nj  # number of states (position and velocity)
nu = 1       # number of control effort input 

# Simulation parameters
dt = 0.01                       # time step (for optimal control problem)
N = 80                          # original OCP horizon
M = 20                          # shorter horizon
K = 50                          # number of initial states for comparisons between 3 MPC formulations

# Cost weights
w_p = 1            # position weight
w_v = 0.1         # velocity weight
w_a = 0.1         # control weight
w_term_cost = 1.0  # terminal cost weight

JMAX = 0           # J_max initialization

# dynamics initialization

f = SimPend(cs.MX.sym('q', nj), cs.MX.sym('dq', nj), cs.MX.sym('u',nu))

# Load the trained neural network model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network model
HiddenLayers = 64
OutputLayers = 1
activation = torch.nn.Tanh()
ModelNN = NeuralNetwork(nx, HiddenLayers, OutputLayers, activation, ub=1)
checkpoint = torch.load("/home/student/shared/orc/Final_Project/Single_Pend/nn_cost_pred.pt", map_location=device, weights_only=True)
ModelNN.load_state_dict(checkpoint["model_state_dict"])
ModelNN.eval()  # Set to evaluation mode
ModelNN.to(device)

# Convert neural network to CasADi function
build_path = os.path.expanduser("/home/student/shared/orc/Final_Project/Single_Pend/_l4c_gen")
os.makedirs(build_path, exist_ok=True)

l4c_model = NNC(ModelNN, device=str(device), build_dir=build_path)
x_sym = MX.sym("x", 1, nx)  # Symbolic input for CasADi


# Arrays to store results
differences = []
x0_vals = []
cmsM = []
cmsMJ = []
cmsNM = []

JM_vec = []
JNM_vec = []

# Loop over K initial states the case 1 and 3 because they do not work with normalization
for _ in range(K):
    x0_val = np.array([np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)])  # Random initial state
    x0_vals.append(x0_val)

    J_optM = OCP_cost(M, x0_val[:1], x0_val[1:], nx, nj, w_p, w_v, w_a, dt, f)
    J_optNM = OCP_cost(M+N, x0_val[:1], x0_val[1:], nx, nj, w_p, w_v, w_a, dt, f) 
    
    # used to find maximum and use it to for the NN that works with normalized costs
    
    JM_vec.append(J_optM)
    JNM_vec.append(J_optNM)
    
    # Calculate average cost per step
    cmsMs = J_optM/M
    cmsM.append(cmsMs)
    cmsNMs = J_optNM/(N+M)
    cmsNM.append(cmsNMs)

JMAX = np.max(JNM_vec)  # Use this for normalization
nn_terminal_cost_norm = Function("nn_terminal_cost", [x_sym], [JMAX*l4c_model(x_sym)])  # Use JMAX not JMAX_M

for _ in range(K):
    
    x0_val = np.array(x0_vals[_])  # Use the same initial state for the terminal cost OCP
    
    # Calculate the terminal cost using the neural network
    ocp_terminal_cost = OCP_Terminal_Cost(
        M, 
        q0=x0_val[:1], 
        dq0=x0_val[1:], 
        terminal_cost_func=nn_terminal_cost_norm, 
        nx  = nx, 
        nu  = nj, 
        w_p = w_p, 
        w_v = w_v,  
        w_a = w_a, 
        dt  = dt, 
        f   = f,
        w_term = w_term_cost,
    )
    
    cmsMJ.append(ocp_terminal_cost/(N+M))  # Average cost per step for the terminal cost OCP
    
# Calculate difference
for i in range(K):
    cmsM_val = cmsM[i]
    cmsMJ_val = cmsMJ[i]
    cmsNM_val = cmsNM[i]
    
    # Calculate normalized difference
    difference_norm = (cmsMJ_val - cmsNM_val) / cmsNM_val  # Normalized difference
    differences.append(difference_norm)

# Plot differences
plt.figure()
plt.plot(range(1, K+1), differences, marker='o', linestyle='-', label='|Terminal Cost - Optimal Cost|/|Optimal Cost|')
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