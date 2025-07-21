import matplotlib.pyplot as plt
from orc.Final_Project.Single_Pend.SinPendTrain import SimPend
from orc.Final_Project.neural_network import NeuralNetwork
import casadi as cs
from orc.Final_Project.OCP_solve import OCP_cost
import torch.nn as nn 
import torch 
import random as rnd
import multiprocessing
import numpy as np
import concurrent.futures
from matplotlib.widgets import Cursor

def compute_costs(args):
    q, v = args
    test_input = torch.tensor([q, v], dtype=torch.float32)
    predicted_cost = ModelNN(test_input).item()  # Remove JMAX multiplication here
    try:
        ocp_cost = OCP_cost(N, [q], [v], nx, nj, w_p, w_v, w_a, dt, f)
    except Exception as e:
        print(f"[WARN] OCP failed for q={q}, v={v}: {e}")
        ocp_cost = 1e6  # Penalizza i fallimenti
    return predicted_cost, ocp_cost

# parameters definition for the pendulum dynamics

g = 9.81                # gravity constant
L = 0.8                 # length of the pendulum
m = 0.25                # mass of the pendulum
b = 0.1                 # damping coefficient
Jp = 1/3* m * L**2      # moment of inertia of the pendulum --> assuming the pendulum is a rod of length L and mass m, pivoted at one end --> COM is at L/2

nj = 1       # number of joints
nx = 2 * nj  # number of states (position and velocity)

# neural network parameters

n_hidden = 64                   # number of neurons for each hidden layers in the neural network
out_size = 1                    # output layer size (cost prediction)
criterion = nn.MSELoss()        # loss function for the neural network
Trainingloops = 2000            # number of training loops for the neural network
activation_func = nn.Tanh()
Learning_rate = 0.7e-3          # learning rate for the Adam optimizer
in_size = nx                    # input layer size (number of states)

ModelNN = NeuralNetwork(in_size,n_hidden,out_size,activation_func,ub=1)

# simulation parameters

dt = 1e-2   # time step
N  = 200    # time horizon

# cost weights

w_p = 1             # position cost weight
w_v = 0.1           # velocity cost weight
w_a = 0.1           # acceleration cost weight

JMAX = 0          # initialization of the saturation cost, it will be updated later with the maximum cost in the dataset 



if __name__=='__main__':
    
    #instantiate the dynamics
    
    f = SimPend(cs.MX.sym('q',nj), cs.MX.sym('dq',nj),cs.MX.sym('u',nj))
    
    # load the trained model 
    
    checkpoint = torch.load('/home/student/shared/orc/Final_Project/Single_Pend/nn_cost_pred.pt')
    
    ModelNN.load_state_dict(checkpoint['model_state_dict'])
    
    # cores for parallelization

    n_cores = multiprocessing.cpu_count() - 4 
    
    # generate a dataset with its cost for comparison
    
    q_vals = np.linspace(-np.pi, np.pi, 30)          # 30 intervals for positions
    v_vals = np.linspace(-np.pi, np.pi, 30)          # 30 intervals for velocities

    # Generate the grid
    grid = [(q, v) for q in q_vals for v in v_vals]

    # Compute costs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(compute_costs, grid))

    # Filter out invalid results (OCP cost >= 1e6)
    valid_data = [(grid_point, result) for grid_point, result in zip(grid, results) if result[1] < 1e6] 
    
    print(f"Total grid points: {len(grid)}")
    print(f"Valid grid points after filtering: {len(valid_data)}")
    print(f"Filtered out: {len(grid) - len(valid_data)} points")

    # Extract filtered data
    valid_grid_points = [data[0] for data in valid_data]  # (q, v) pairs
    predicted_costs_filtered = [data[1][0] for data in valid_data]  # predicted costs
    ocp_costs_filtered = [data[1][1] for data in valid_data]  # OCP costs

    # Create mapping from grid coordinates to filtered costs for plotting
    # Initialize matrices with NaN for invalid points
    costs_pred = np.full((len(q_vals), len(v_vals)), np.nan)
    costs_ocp = np.full((len(q_vals), len(v_vals)), np.nan)

    # Fill in valid costs
    for (q, v), (pred_cost, ocp_cost) in valid_data:
        # Find indices in the original grid
        i = np.argmin(np.abs(q_vals - q))
        j = np.argmin(np.abs(v_vals - v))
        costs_pred[i, j] = pred_cost
        costs_ocp[i, j] = ocp_cost
        
    # Find JMAX from valid costs only
    JMAX = np.nanmax(costs_ocp)
    print(f"JMAX (maximum valid OCP cost): {JMAX}")

    # Denormalize predicted costs BEFORE any plotting
    costs_pred = costs_pred
    costs_ocp_n = costs_ocp / JMAX  # normalized costs

    # Filter out unreasonable costs
    reasonable_mask = np.logical_and(~np.isnan(costs_ocp), costs_ocp < 1e6)
    reasonable_costs = costs_ocp[reasonable_mask]
    reasonable_preds = costs_pred[reasonable_mask]  # Now using denormalized predictions

    # First set of plots (heatmaps)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(costs_pred, extent=[v_vals[0], v_vals[-1], q_vals[0], q_vals[-1]], 
                     origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im1, label="Predicted Cost")
    plt.xlabel("Initial Velocity (rad/s)")
    plt.ylabel("Initial Position (rad)")
    plt.title("Predicted Cost (Filtered)")

    plt.subplot(1, 3, 2)
    im2 = plt.imshow(costs_ocp, extent=[v_vals[0], v_vals[-1], q_vals[0], q_vals[-1]], 
                     origin='lower', aspect='auto', cmap='viridis', vmin=0., vmax=900.)
    plt.colorbar(im2, label="Optimal Cost by OCP")
    plt.xlabel("Initial Velocity (rad/s)")
    plt.ylabel("Initial Position (rad)")
    plt.title("Optimal Cost (Filtered)")
    
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(costs_ocp_n, extent=[v_vals[0], v_vals[-1], q_vals[0], q_vals[-1]], 
                     origin='lower', aspect='auto', cmap='viridis', vmin=0., vmax=1.)
    plt.colorbar(im3, label="Normalized Optimal Cost")
    plt.xlabel("Initial Velocity (rad/s)")
    plt.ylabel("Initial Position (rad)")
    plt.title("Normalized Optimal Cost (Filtered)")
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    if len(ocp_costs_filtered) > 0:
        print(f"\nStatistics of valid OCP costs:")
        print(f"  Min: {np.min(ocp_costs_filtered):.4f}")
        print(f"  Max: {np.max(ocp_costs_filtered):.4f}")
        print(f"  Mean: {np.mean(ocp_costs_filtered):.4f}")
        print(f"  Std: {np.std(ocp_costs_filtered):.4f}")
    
    # Filter out unreasonable costs (similar to double pendulum)
    reasonable_mask = np.logical_and(~np.isnan(costs_ocp), costs_ocp < 1e6)
    reasonable_costs = costs_ocp[reasonable_mask]
    reasonable_preds = costs_pred[reasonable_mask]*JMAX  # Denormalize predictions using JMAX

    # Scatter plot with filtered data
    plt.figure(figsize=(8,6))
    plt.scatter(reasonable_preds, reasonable_costs, alpha=0.7, label='Predictions VS computed OCP cost')
    if len(reasonable_costs) > 0:
        min_val = min(min(reasonable_preds), min(reasonable_costs))
        max_val = max(max(reasonable_preds), max(reasonable_costs))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    ax = plt.gca()
    plt.title("Prediction Model VS OCP solver")
    plt.xlabel("Predicted cost")
    plt.ylabel("True cost (OCP)")
    plt.legend()
    plt.grid(True)
    Cursor(ax, useblit=True, color='gray', linewidth=0.5)
    plt.show()

    # Residual plot
    residuals = (reasonable_preds - reasonable_costs)/ reasonable_costs  # Relative residuals
    predictions_arr = reasonable_preds
    residuals_arr = residuals

    plt.figure(figsize=(10, 6))
    plt.scatter(predictions_arr, residuals_arr, alpha=0.7,label=' normalized Residuals')
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
    plt.title("Normalized Residuals of the Predictions (Filtered)")
    plt.xlabel("Predicted Cost")
    plt.ylabel("Normalized Residual (Predicted - True Cost)")
    plt.legend()
    plt.grid(True)
    plt.show()