from orc.Final_Project.neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from adam.casadi.computations import KinDynComputations
import casadi as cs
from casadi import Function
from l4casadi import L4CasADi as NNC
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

# Import the solutions
from orc.Final_Project.OCP_solve import OCP_cost

# Parameters for the single pendulum
g = 9.81        
L = 0.8         
m = 0.25         
b = 0.1
Jp = 1/3* m * L**2      # moment of inertia of the pendulum --> assuming the pendulum is a rod of length L and mass m, pivoted at one end --> COM is at L/2

nq = 1    # number of joints
nx = 2*nq # size of the state variable

# Training parameters
HiddenLayers = 64
OutputLayers = 1
TrainingLoops = 2000 #(full-batch)
activation = nn.Tanh()
LearningRate = 0.7e-3

# Simulation parameters
N = 200  # Aumentato come double pendulum
dt = 0.01 # time step for the simulation

# Cost weights
w_p = 1      # position weight
w_v = 1e-1      # velocity weight
w_a = 1e-1      # acceleration weight
JMAX = 0

# State and input variables
def SimPend(q, dq, u):
    """Defines the dynamics of a simple pendulum."""
    state = cs.vertcat(q, dq)
    ddq = Jp**-1 * (u-m*g*L/2 * cs.sin(q)-b*dq)
    dstate = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, u], [dstate])
    return f

def compute_cost_for_state(state_pair):
    """
    Compute optimal cost for a single (q0, dq0) pair.
    This function will be called by each worker process.
    """
    q0, dq0 = state_pair
    
    # Create the dynamics function (each process needs its own)
    f = SimPend(cs.MX.sym('q', nq), cs.MX.sym('dq', nq), cs.MX.sym('u'))
    
    q0_array = np.array([q0])
    dq0_array = np.array([dq0])
    
    # Compute the optimal cost numerically
    J_opt = OCP_cost(N, q0_array, dq0_array, nx, nq, w_p, w_v, w_a, dt, f)
    
    return [np.concatenate([q0_array, dq0_array]), J_opt]

def main():
    
    # Determine number of cores to use
    total_cores = mp.cpu_count()
    cores_to_use = max(1, total_cores - 4)  # Use cpu_count-4, but at least 1
    print(f"Total CPU cores: {total_cores}")
    print(f"Using {cores_to_use} cores for parallel computation")

    # Create the dataset for storing initial state and cost
    DataSet = []
    print("Start solving the optimization problems")

    # Discretize angular position and angular velocity into 30 intervals
    q_vals = np.linspace(-np.pi, np.pi, 60)  # Angular position
    dq_vals = np.linspace(-np.pi, np.pi, 60)    # Angular velocity

    # Create all state pairs to be computed
    state_pairs = [(q0, dq0) for q0 in q_vals for dq0 in dq_vals]
    total_computations = len(state_pairs)
    print(f"Total computations to perform: {total_computations}")

    # Parallel computation using multiprocessing
    print("Starting parallel cost computation...")
    
    with Pool(processes=cores_to_use) as pool:
        # Use imap to get results as they complete (with progress tracking)
        results = []
        for i, result in enumerate(pool.imap(compute_cost_for_state, state_pairs)):
            results.append(result)
            
            # Print progress every 100 computations
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{total_computations} computations ({(i+1)/total_computations*100:.1f}%)")
    
    DataSet = results
    print("Parallel computation completed!")

    # Shuffle the dataset randomly
    random.shuffle(DataSet)

    print("DataSet built properly!")

    cost_values = np.array([data[1] for data in DataSet])
    plt.hist(cost_values, bins=50)
    plt.xlabel("Cost Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cost Values in Dataset")
    plt.show()

    J_vals = [data[1] for data in DataSet]
    JMAX = max(J_vals)

    # Prepare the training dataset
    x_train = torch.tensor(np.array([data[0] for data in DataSet]), dtype=torch.float32)
    y_train = torch.tensor(np.array([data[1]/JMAX for data in DataSet]), dtype=torch.float32).unsqueeze(1)

    # Define the neural network model
    ModelNN = NeuralNetwork(nx, HiddenLayers, OutputLayers, activation, ub=1)

    weights = torch.tensor(
        [1.0 if data[1] < 200 else 2.5 if data[1] < 250 else 10.0 for data in DataSet],
        dtype=torch.float32
    ).unsqueeze(1)

    # Optimizer and loss function
    optimizer = optim.Adam(ModelNN.parameters(), LearningRate)
    loss_fn = nn.MSELoss()

    # Training loop
    losses = []

    print("Starting training loop...")
    for loop in range(TrainingLoops):
        # Training step
        optimizer.zero_grad()
        predictions = ModelNN(x_train)
        loss = (loss_fn(predictions, y_train)* weights).mean()
        loss.backward()
        optimizer.step()

        # Store losses
        losses.append(loss.item())

        if loop % 50 == 0:
            print(f"Loop {loop}/{TrainingLoops}, Loss: {loss.item()}")

    # Save the trained model
    # torch.save({'model_state_dict': ModelNN.state_dict()}, '/home/student/shared/orc/Final_Project/Single_Pend/nn_cost_pred.pt')
    # print("Model correctly saved as 'nn_cost_pred.pt'")

    # Plot the training loss
    plt.plot(range(TrainingLoops), losses, label="Training Loss")
    plt.xlabel("Training Loops")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
    
if __name__=='__main__':
    main()