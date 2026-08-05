import torch
import torch.nn as nn
import os
import multiprocessing
import casadi as cs
import pinocchio as pin
import numpy as np
from orc.Final_Project.neural_network import NeuralNetwork
import orc.Final_Project.OCP_solve as OCP
from example_robot_data.robots_loader import load
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.robot_wrapper import RobotWrapper
import orc.Final_Project.Double_Pend.DP_conf as conf
import random as rnd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Cursor
try:
    matplotlib.use('TkAgg') # set how the backend should work
except ImportError:
    matplotlib.use('Agg') # headless environment (no tk/display available)
import time

"""This script trains a neural network to control a double pendulum using an optimal control problem (OCP) formulation."""

robot = load("double_pendulum")

robot.na = robot.nv # set the number of actuated joints --> RobotSimulator won't work otherways 

#data = robot.createData()

nj = 2  # 2 joints 

nx = 2*nj 

# required parameters parsing from urdf file  

params_pend = {
    
    'nq': nj,                                                     # number of coordinates
    'L1': robot.model.jointPlacements[1].translation[2],          # length first pendulum --> z component taken as the most relevant--> frame definition reminds an inverted D.H. notation x for joint axis and z for linkage axes 
    'L2': robot.model.jointPlacements[2].translation[2],          # length second pendulum
    'm1': robot.model.inertias[1].mass,                           # mass first pendulum
    'm2': robot.model.inertias[2].mass,                           # mass second pendulum
    'g' : 9.81,                                                   # gravity constant
    'J1': robot.model.inertias[1].inertia[0,0],                   # inertia first pendulum (Ixx) --> COM referenced inertial tensor component
    'J2': robot.model.inertias[2].inertia[0,0],                   # inertia second pendulum (Ixx)
    'b1': 0.05,                                                   # read from the urdf <dynamics>...</dynamics> --> reducing it gives a nice oscillatory behaviour, the damping here is quite high
    'b2': 0.05,
    'COM1': robot.model.inertias[1].lever[2],                     # position of COM  w.r.t joint frame 
    'COM2': robot.model.inertias[2].lever[2]                      # use only z-component --> same reason as for L1 and L2 --> z effective component for Hyugens-Steiner inertial term
}


class Pendulum :
    def __init__(self,params):
        
        self.N_link = params['nq']
        
        self.L1 = params['L1']
        
        self.L2 = params['L2']
        
        self.m1 = params['m1']
        
        self.m2 = params['m2']
        
        self.J1 = params['J1'] 
        
        self.J2 = params['J2']
        
        self.b1 = params['b1']
        
        self.b2 = params['b2']
        
        self.g = params['g']
        
        self.com1 = params['COM1'] 
        
        self.com2 = params['COM2']
    
    def dynamics(self,q,dq,u):
        
        """ Dynamics of the double pendulum in state-space model, the equations were computed thanks to Euler-Lagrange 
        approach in Maple to compute the matrices A and b entries """
        
        q = cs.vec(q)
        dq = cs.vec(dq)
        u = cs.vec(u)
        
        # state-sapce model
        
        state = cs.vertcat(q,dq)    # [q1,q2,dq1,dq2]^T
        
        # Parameters
        m1, m2 = self.m1, self.m2
        L1, L2 = self.L1, self.L2
        J1,J2 = self.J1 + self.m1*self.com1**2 ,self.J2 + self.m2*self.com2**2   # adding Hyugens Steiner terms 
        g = self.g
        b1,b2 = self.b1,self.b2
        COM1,COM2 = self.com1,self.com2
        
        # Matrix A (4x4)
        A = cs.MX.zeros(4, 4)
        
        # Fill A matrix (upper part - dynamics)
        
        A[0,0] = -1
        A[1,1] = -1
        A[2,0] = b1
        A[2,2] = 2 * cs.cos(q[1]) * m2 * L1 * COM2 + (COM2**2 + L1**2) * m2 + (m1 * COM1**2) + J1
        A[2,3] = m2*COM2*(cs.cos(q[1])*L1+COM2)
        A[3,1] = b2
        A[3,2] = m2*COM2*(cs.cos(q[1])*L1 + COM2)
        A[3,3] = J2 + m2*COM2**2
                       
        # Vector b (4x1)
        b = cs.MX.zeros(4, 1)
        
        # First dynamics equation
        b[0] = -dq[0]
        
        # Second dynamics equation
        b[1] = -dq[1]
        
        # Kinematic relations between position and velocities in the state 
        
        b[2] = (
            COM2*L1*dq[1]**2*m2*cs.sin(q[1]) + 
            2*COM2*L1*dq[0]*dq[1]*m2*cs.sin(q[1]) - 
            g*((cs.cos(q[1])*m2*COM2 + m1*COM1 + 
            L1*m2)*cs.cos(q[0]) - cs.sin(q[0])*cs.sin(q[1])*m2*COM2) + 
            u[0]
        )
        b[3] = (
            -COM2*L1*m2*cs.sin(q[1])*dq[0]**2 - 
            m2*g*(cs.cos(q[0])*cs.cos(q[1]) 
            - cs.sin(q[0])*cs.sin(q[1]))*COM2 + 
            u[1]
        ) 
        
        # Solve for accelerations
        
        reg = 1e-6 # moves away from singular configurations of A --> Sylvester method --> useless for the most part :-)
        
        A_reg = A + reg * cs.MX.eye(4)
        
        ddq_sol = cs.solve(A_reg, b)
        
        # Return state derivatives
        
        dstate = ddq_sol
                
        f = cs.Function('f',[state,u],[dstate])
         
        return f

# define the expanded NN to predict the cost of the OCPs --> complex dynamics, so we need a complex NN to predict the cost of the OCPs

class ExtendedNeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, input_size, hidden_size, output_size,activation=nn.Tanh(), ub=None):
        super().__init__()  # Initialize the parent class
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size,hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
        )
        self.ub = ub if ub is not None else 1 # upper bound of the output layer
        self.initialize_weights()

    def forward(self, x):
        out = self.linear_stack(x) * self.ub
        return out
   
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias) 

def train_NN(model, input_tensor, target_tensor, criterion, num_epochs, lr, verbose=True):
    """
    Trains the given neural network model using full-batch gradient descent and plots training loss.

    Parameters:
    - model: instance of ExtendedNeuralNetwork
    - input_tensor: torch.Tensor of shape (N, input_dim)
    - target_tensor: torch.Tensor of shape (N, 1)
    - criterion: loss function (e.g., nn.MSELoss())
    - num_epochs: number of training epochs
    - lr: learning rate for optimizer
    - verbose: whether to print training progress
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor).mean()  # Compute mean loss over the batch
        loss.backward()
        optimizer.step()

        mean_loss = loss.item()
        losses.append(mean_loss)

        if verbose and (epoch % 100 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {mean_loss:.6f}")

    print("Training complete.")

    # Plotting the loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss Over Epochs (Full Batch)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return optimizer, num_epochs, mean_loss  # Return optimizer state, last epoch, and final loss
#define the number of OCPs to solve

N_OCPs = 3000  # number of OCPs to solve --> defines how many initial states we have to generate for the dataset

# neural network parameters

n_hidden = 128                  # number of neurons in each hidden layers in the neural network
out_size = 1                    # output layer size (cost prediction)
criterion = nn.MSELoss()        # loss function for the neural network
Trainingloops = 1500            # number of training loops for the neural network
activation_func = nn.Tanh()
Learning_rate = 0.009           # learning rate for the Adam optimizer
in_size = nx

# simulation parameters

dt = 1e-2    # time step
Ns  = 200    # time horizon for cost prediction 

N = 500      # time samples for the trajectory plotting test for the dynamics 

# cost weights

w_p = 1             # position cost weight
w_v = 0.05          # velocity cost weight
w_a = 0.03          # acceleration cost weight

# use parallelism 

n_cores = multiprocessing.cpu_count() - 4  

def generate_single_sample(args):
    """ Generate a single OCP data point """
    N, nx, nj, w_p, w_v, w_a, dt, f = args['N'], args['nx'], args['nj'], args['w_p'], args['w_v'], args['w_a'], args['dt'], args['f']
    
    def generate_NS_IC():
        q  = [rnd.uniform(-1, 1) for _ in range(nj)]
        dq = [rnd.uniform(-2., 2.) for _ in range(nj)]
        return q, dq

    q0, dq0 = generate_NS_IC()
    q0 = np.array([qi * np.pi for qi in q0]).reshape((nj,1))
    dq0 = np.array([dqi * np.pi for dqi in dq0]).reshape((nj,1))
    state_0 = np.concatenate([q0,dq0])
    
    try:
        cost = OCP.OCP_cost(Ns, q0, dq0, nx, nj, w_p, w_v, w_a, dt, f)
    except Exception as e:
        print(f"[WARN] OCP failed for q0={q0}, dq0={dq0}: {e}")
        cost = 1e6  # set very high in order to tell the NN during training that SC as initial state are very 'unlikely' or better non desirable, so the predicted cost will be outrageously high !!! 
    return state_0, cost

class Dataset : 
    
    def __init__(self, Ns, N_OCPs, dt, f, w_p, w_v, w_a, J_SAT, n_workers):
        self.N = Ns
        self.n_workers = n_workers
        self.N_OCPs = N_OCPs
        self.w_a = w_a
        self.w_p = w_p
        self.w_v = w_v
        self.J_SAT = J_SAT
        self.dt = dt
        self.f = f
        self.generate_dataset()
        
    def generate_dataset(self):
        """ Generate the dataset by solving OCPs in parallel """
        
        args = {
            'N': self.N,
            'nx': 2*params_pend['nq'],
            'nj': params_pend['nq'],
            'w_p': self.w_p,
            'w_v': self.w_v,
            'w_a': self.w_a,
            'dt': self.dt,
            'f': self.f
        }
        
        with multiprocessing.Pool(self.n_workers) as pool:
            results = pool.map(generate_single_sample, [args] * self.N_OCPs)
        
        # Unzip the results into states and costs
        states, costs = zip(*results)
        
        # Convert to numpy arrays
        self.states = np.array(states).squeeze(-1)
        self.costs = np.array(costs)
        
        # eliminate the outliers from the dataset
        mask =  self.costs < 1e6  # filter out costs that are too high (indicating failed OCPs), they  
        self.states = self.states[mask]
        self.costs = self.costs[mask]
        # normalize costs
        
        self.J_SAT = self.J_max_compute()
        self.costs_n = self.costs / self.J_SAT
        
        # compute time-average costs --> needed to compare different time horizon choices 
        
        self.costs_avg = self.costs/Ns
    
    def J_max_compute(self):
        """ Compute the maximum cost in the dataset """
        self.costs = np.array(self.costs)  # Ensure costs is a numpy array
        if len(self.costs) == 0:
            raise ValueError("No costs available to compute J_max.")
        # Return the maximum cost    
        return np.max(self.costs)
    
def simulate_trajectory(f, q0, dq0, N, dt, nu): # natural response  
    """
    Simulate the pendulum trajectory from initial state using zero control.
    Returns arrays of q, dq over time.
    """
    q0 = np.array(q0).flatten() # convert to column vector 
    dq0 = np.array(dq0).flatten() 
    q = [q0.copy()]
    dq = [dq0.copy()]
    state = np.concatenate([q0, dq0])
    u = np.zeros(nu) # Zero control input
    for _ in range(N):
        i = 0  
        # convert state into Casadi var
        state_dm = cs.DM(state.flatten())
        
        u_dm = cs.DM(u.flatten())
       
        state_dot = np.array(f(state_dm, u_dm).full()).flatten()
       
        state = state + dt * state_dot
       
        q.append(state[:nj].copy())
      
        dq.append(state[nj:].copy())
       
        i+=1
    
    return np.array(q), np.array(dq)

def plot_simulated_trajectory(dataset, f, dt, N, sample_idx=0):
    
    q0 = dataset.states[sample_idx, :nj]
    dq0 = dataset.states[sample_idx, nj:]
    q_traj, dq_traj = simulate_trajectory(f, q0, dq0, N, dt, nj)
    time = np.arange(N+1) * dt
    ddq_traj = np.gradient(dq_traj,dt,axis=0)
        
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(time, q_traj[:,0], label='q1 [rad]')
    axs[0].plot(time, q_traj[:,1], label='q2 [rad]')
    axs[0].set_ylabel('Position [rad]')
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(time, dq_traj[:,0], label='dq1 [rad/s]', color ='orange')
    axs[1].plot(time, dq_traj[:,1], label='dq2 [rad/s]', color ='blue')
    axs[1].set_ylabel('Velocity [rad/s]')
    axs[1].grid(True)
    axs[1].legend()
    axs[2].plot(time, ddq_traj[:,0], label='ddq1 [rad/s²]', color='green')
    axs[2].plot(time, ddq_traj[:,1], label='ddq2 [rad/s²]', color='red')
    axs[2].set_ylabel('Accelerazione [rad/s²]')
    axs[2].set_xlabel('Tempo [s]')
    axs[2].grid(True)
    axs[2].legend()
    fig.suptitle(f"Simulated trajectory from sample #{sample_idx}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Aggiungi cursori a ogni subplot
    
    cursors = [Cursor(ax, useblit=True, color='gray', linewidth=0.5) for ax in axs]

    plt.show()

    
    
if __name__ == "__main__":
    
    rnd.seed(11)  # Seed the random number generator for reproducibility

    #Create and launch the simulator 

    sim = RobotSimulator(conf, robot)

    time.sleep(2)   # safety load time for the simulator 
    
    pend_wrapper = Pendulum(params_pend) # used to define the pendulum dynamics
    
    f = pend_wrapper.dynamics(cs.MX.sym("q",nj,1),cs.MX.sym("dq",nj,1),cs.MX.sym("u",nj,1))
    
    J_SAT = [] # init the J_SAT value 
    
    args_prova = {
            'N': Ns,
            'nx': params_pend['nq'],
            'nj': params_pend['nq'],
            'w_p': w_p,
            'w_v': w_v,
            'w_a': w_a,
            'dt': dt,
            'f': f
        }
    
    dataset = Dataset(Ns, N_OCPs, dt, f, w_p, w_v, w_a, J_SAT, n_cores)

    # Test the dynamics on the genrated initial states for the pendulum --> natural response (zero input) will be returned 
    
    trajectories = simulate_trajectory(f,dataset.states[1,:nj],dataset.states[1,nj:],N,dt,nu=nj)    
        
    plot_simulated_trajectory(dataset, f, dt, N, sample_idx=0)
    
    offset = [] # needed because the frame of the simulator has a phase offset of 90° compared to the one I used for the dynamics
    
    for _ in trajectories[0]: # compute offset trajectory 
        
        delta = np.array([-np.pi/2,0]).flatten()
        
        offset.append(delta) 
    
    sim.display_motion(trajectories[0]+offset,dt) # now goes to stable equilibrium position as we expect 
     
    # Convert dataset to PyTorch tensors
    
    initial_states_tensor = torch.tensor(dataset.states, dtype=torch.float32)
    optimal_costs_tensor = torch.tensor(dataset.costs_n, dtype=torch.float32)

    # Plot cost distribution
    plt.figure(figsize=(8, 6))
    plt.hist(dataset.costs_n, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Normalized Optimal Costs')
    plt.xlabel('Normalized Cost [.]')
    plt.ylabel('Frequency [.]')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.hist(dataset.costs, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Optimal Costs')
    plt.xlabel('Cost [.]')
    plt.ylabel('Frequency [.]')
    plt.grid(True)
    plt.show()
    
    # instantiate our NN model for the function approximation problem 
    
    print("\n [INFO] Initializing the neural network for cost prediction...\n")

    nn_cost_pred = ExtendedNeuralNetwork(in_size, n_hidden, out_size,activation=activation_func, ub=1.0)

    print("[INFO] Neural network instantiated with architecture:\n")
    
    print(f"Input size: {in_size}, Hidden size: {n_hidden}, Output size: {out_size}\n")

    # Train the neural network
    
    NN_out_data = train_NN(nn_cost_pred, initial_states_tensor, optimal_costs_tensor, criterion, Trainingloops, Learning_rate)
    
    # save the model 
    
    # torch.save({
    #     'model_state_dict': nn_cost_pred.state_dict(),        ## weights of the model
    #     'optimizer_state_dict': NN_out_data[0].state_dict(),  ## optimizer state
    #     'epoch': NN_out_data[1],                              ## number of the epoch in which the model has been saved 
    #     'loss': NN_out_data[2],                               ## loss of the last epoch 
    # }, '/home/student/shared/orc/Final_Project/Double_Pend/nn_cost_pred.pt')
    
    print("[INFO] Neural network training completed and model saved as 'nn_cost_pred.pt'.\n")

    print("[INFO] Exiting the script...\n")
    
    print("[INFO] Script execution completed successfully.\n")
    
    
    

    
    
    
                
            
