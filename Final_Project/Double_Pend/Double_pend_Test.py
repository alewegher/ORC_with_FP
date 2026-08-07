import os
import matplotlib.pyplot as plt
from orc.Final_Project.Double_Pend.Double_pend_training import Pendulum
from orc.Final_Project.Double_Pend.Double_pend_training import ExtendedNeuralNetwork
import casadi as cs
from orc.Final_Project.OCP_solve import OCP_cost
import torch.nn as nn 
import torch 
import random as rnd
import multiprocessing
import numpy as np
import concurrent.futures
from example_robot_data.robots_loader import load
import math 
from matplotlib.widgets import Cursor


def compute_ocp_parallel(args):
    """
    Compute OCP cost for a single initial condition
    Returns 1e6 if solver fails (same as training)
    """
    i, nj, N, nx, w_p, w_v, w_a, dt, params_pend = args
    
    # Recreate pendulum and dynamics in each process
    double_pend = Pendulum(params=params_pend)
    f = double_pend.dynamics(cs.MX.sym('q',nj),cs.MX.sym('dq',nj),cs.MX.sym('u',nj))
    
    # Generate random initial condition
    q0 = np.random.uniform(low=-np.pi, high=np.pi, size=nj)
    dq0 = np.random.uniform(low=-2.*np.pi, high=2.*np.pi, size=nj)

    try:
        J_comp = OCP_cost(N, q0, dq0, nx, nj, w_p, w_v, w_a, dt, f)
    except:
        J_comp = 1e6  # Same penalty as training
    
    # Create input tensor for NN
    x_tens = torch.tensor(np.concatenate([q0, dq0]), dtype=torch.float32)
    
    return J_comp, x_tens


robot = load("double_pendulum")
robot.na = robot.nv

nj = 2
nx = 2*nj

# Parameter parsing from urdf to build Pendulum 
params_pend = {
    'nq': nj,
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

# Simulation parameters
dt = 0.01
N = 200
N_OCPs = 3000

# Cost weights
w_p = 1
w_v = 0.05
w_a = 0.03

JMAX = 0

# Neural network parameters
n_hidden = 128
out_size = 1
activation_func = nn.Tanh()
in_size = nx

# Parallelism
n_cores = multiprocessing.cpu_count() - 5

# Instantiate the NN
NN_Model = ExtendedNeuralNetwork(in_size, n_hidden, out_size, activation_func, ub=1.)


if __name__=='__main__':
    
    # Load the trained model
    checkpoint = torch.load('/home/student/shared/orc/Final_Project/Double_Pend/nn_cost_pred.pt')
    NN_Model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare arguments for parallel processing
    args_list = [(i, nj, N, nx, w_p, w_v, w_a, dt, params_pend) for i in range(N_OCPs)]
    
    # Parallel computation
    print(f"Computing {N_OCPs} OCPs using {n_cores} cores...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(compute_ocp_parallel, args_list))
    
    # Extract results
    mask = [result[0] < 1e6 for result in results]  # Mask for valid costs
    true_cost = [result[0] for result in results]
    x_tensors = [result[1] for result in results]
    
    # Compute NN predictions
    predictions_norm = []
    for x_tens in x_tensors:
        y_pred = NN_Model(x_tens).item()
        predictions_norm.append(y_pred)
    
    # Denormalize predictions
    JMAX = max(cost for cost in true_cost if cost < 1e6)  # Don't use penalty costs for normalization
    if JMAX == 0:
        JMAX = 1.0  # Fallback
        
    predictions = [JMAX * pred for pred in predictions_norm]
    
    # Only plot y=x line for reasonable costs (not penalty costs)
    reasonable_costs = [cost for cost in true_cost if cost < 1e6]
    reasonable_preds = [pred for pred, cost in zip(predictions, true_cost) if cost < 1e6]
    
    
    FIGURES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "saved_images"))
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Scatter plot
    fig_scatter = plt.figure(figsize=(8,6))
    plt.scatter(reasonable_preds,reasonable_costs, alpha=0.7, label='Predictions VS computed OCP cost')
    if reasonable_costs:
        min_val = min(min(reasonable_preds), min(reasonable_costs))
        max_val = max(max(reasonable_preds), max(reasonable_costs))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    ax=plt.gca()
    plt.title("Prediction Model VS OCP solver")
    plt.xlabel("Predicted cost")
    plt.ylabel("True cost (OCP)")
    plt.legend()
    plt.grid(True)
    Cursor(ax,useblit=True, color='gray', linewidth=0.5)
    fig_scatter.savefig(os.path.join(FIGURES_DIR, "DP_scatter.png"), dpi=150, bbox_inches='tight')
    plt.show()

    # residual plot senza filtri
    residuals = [pred - cost for pred, cost in zip(reasonable_preds, reasonable_costs)]
    predictions_arr = np.array(reasonable_preds)
    residuals_arr = np.array(residuals)

    # Ordina predictions e residuals per predictions crescenti
    sorted_indices = np.argsort(predictions_arr)
    predictions_sorted = predictions_arr[sorted_indices]
    residuals_sorted = residuals_arr[sorted_indices]

    # plot residuals
    fig_residuals = plt.figure(figsize=(8,6))
    plt.scatter(predictions_sorted, residuals_sorted, alpha=0.7, label='Residuals')
    plt.axhline(0, color='red', linestyle='--', label='Zero line')
    plt.xlim(left=0, right=max(predictions_sorted) * 1.1)
    plt.ylim(bottom=min(residuals_sorted)*1.1, top=max(residuals_sorted)*1.1)
    ax = plt.gca()
    plt.xticks(np.arange(0, max(predictions_sorted) * 1.1, step=max(predictions_sorted) * 0.1))
    plt.yticks(np.arange(min(residuals_sorted) * 1.1, max(residuals_sorted) * 1.1, step=(max(residuals_sorted) - min(residuals_sorted)) * 0.1))
    plt.title("Residuals of Predictions")
    plt.xlabel("Predicted cost")
    plt.ylabel("Residual (Predicted - True cost)")
    plt.legend()
    plt.grid(True)
    Cursor(plt.gca(), useblit=True, color='gray', linewidth=0.5)
    plt.tight_layout()
    fig_residuals.savefig(os.path.join(FIGURES_DIR, "DP_residuals.png"), dpi=150, bbox_inches='tight')
    plt.show()
