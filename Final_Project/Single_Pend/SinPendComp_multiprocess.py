import os
import numpy as np
import casadi as cs
from casadi import Function, MX
import torch
from orc.Final_Project.neural_network import NeuralNetwork
from l4casadi import L4CasADi as NNC
from orc.Final_Project.OCP_solve import OCP_Terminal_Cost, OCP_cost
from orc.Final_Project.Single_Pend.SinPendTrain import SimPend
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
from functools import partial
import time
import tempfile
import shutil

# parameters definition for the pendulum dynamics
g = 9.81                # gravity constant
L = 0.8                 # length of the pendulum
m = 0.25                # mass of the pendulum
b = 0.1                 # damping coefficient
Jp = 1/3* m * L**2      # moment of inertia of the pendulum

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

# Number of cores to use
ncores = mp.cpu_count() - 4

def compute_initial_costs(args):
    """
    Phase 1: Compute J_optM and J_optNM for a single initial state
    This is used to find JMAX and JMAX_M
    """
    x0_val, M, N, nx, nj, w_p, w_v, w_a, dt = args
    
    try:
        # Initialize dynamics (this needs to be done in each process)
        f = SimPend(cs.MX.sym('q', nj), cs.MX.sym('dq', nj), cs.MX.sym('u', nu))
        
        J_optM = OCP_cost(M, x0_val[:1], x0_val[1:], nx, nj, w_p, w_v, w_a, dt, f)
        J_optNM = OCP_cost(M+N, x0_val[:1], x0_val[1:], nx, nj, w_p, w_v, w_a, dt, f)
        
        return {
            'x0_val': x0_val,
            'J_optM': J_optM,
            'J_optNM': J_optNM
        }
        
    except Exception as e:
        print(f"Error in phase 1 for initial state {x0_val}: {e}")
        return None

def setup_neural_network_for_process(process_id, JMAX):  # Changed from JMAX_M to JMAX
    """Setup neural network with process-specific build directory and correct scaling"""
    device = torch.device("cpu")  # Use CPU to avoid CUDA context issues
    
    # Define the neural network model
    HiddenLayers = 64
    OutputLayers = 1
    activation = torch.nn.Tanh()
    ModelNN = NeuralNetwork(nx, HiddenLayers, OutputLayers, activation, ub=1)
    checkpoint = torch.load("/home/student/shared/orc/Final_Project/Single_Pend/nn_cost_pred.pt", 
                           map_location=device, weights_only=True)
    ModelNN.load_state_dict(checkpoint["model_state_dict"])
    ModelNN.eval()
    ModelNN.to(device)
    
    # Create process-specific build directory
    base_build_path = "/home/student/shared/orc/Final_Project/Single_Pend"
    build_path = os.path.join(base_build_path, f"_l4c_gen_proc_{process_id}_{os.getpid()}")
    
    # Clean up existing directory if it exists
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    os.makedirs(build_path, exist_ok=True)
    
    try:
        # Convert neural network to CasADi function with process-specific path
        l4c_model = NNC(ModelNN, device=str(device), build_dir=build_path)
        x_sym = MX.sym("x", 1, nx)
        
        # IMPORTANT: Use JMAX instead of JMAX_M for correct scaling
        nn_terminal_cost_norm = Function("nn_terminal_cost", [x_sym], [JMAX * l4c_model(x_sym)])
        
        return nn_terminal_cost_norm, build_path
        
    except Exception as e:
        print(f"Error setting up neural network for process {process_id}: {e}")
        if os.path.exists(build_path):
            shutil.rmtree(build_path)
        raise e

def compute_terminal_cost(args):
    """Phase 2: Compute OCP with terminal cost for a single initial state"""
    x0_val, J_optM, J_optNM, M, N, nx, nj, w_p, w_v, w_a, dt, w_term_cost, JMAX, process_id = args
    
    build_path = None
    try:
        f = SimPend(cs.MX.sym('q', nj), cs.MX.sym('dq', nj), cs.MX.sym('u', nu))
        nn_terminal_cost_norm, build_path = setup_neural_network_for_process(process_id, JMAX)
        
        ocp_terminal_cost = OCP_Terminal_Cost(
            M, 
            q0=x0_val[:1], 
            dq0=x0_val[1:], 
            terminal_cost_func=nn_terminal_cost_norm, 
            nx=nx, 
            nu=nj, 
            w_p=w_p, 
            w_v=w_v,  
            w_a=w_a, 
            dt=dt, 
            f=f,
            w_term=w_term_cost,
        )
        
        # Calcolo corretto dei costi medi per step
        cmsMs = J_optM / M
        cmsMJs = ocp_terminal_cost / (N+M)  # Normalizzato per N+M come nel single core
        cmsNMs = J_optNM / (N+M)
        
        # Calcolo corretto della differenza normalizzata
        difference_norm = (cmsMJs - cmsNMs) / cmsNMs
        
        result = {
            'x0_val': x0_val,
            'ocp_terminal_cost': ocp_terminal_cost,
            'J_optM': J_optM,
            'J_optNM': J_optNM,
            'difference': difference_norm,
            'cmsMs': cmsMs,
            'cmsMJs': cmsMJs,
            'cmsNMs': cmsNMs,
            'process_id': process_id
        }
        
        return result
        
    except Exception as e:
        print(f"Error in phase 2 for initial state {x0_val} in process {process_id}: {e}")
        return None
        
    finally:
        # Clean up process-specific build directory
        if build_path and os.path.exists(build_path):
            try:
                shutil.rmtree(build_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up {build_path}: {cleanup_error}")

def compute_batch_terminal_cost(args):
    """Phase 2 batch version: compute terminal costs for a batch of initial states"""
    batch_data, JMAX, batch_id = args  # Changed JMAX_M to JMAX
    
    build_path = None
    results = []
    
    try:
        # Initialize dynamics once for the batch
        f = SimPend(cs.MX.sym('q', nj), cs.MX.sym('dq', nj), cs.MX.sym('u', nu))
        
        # Setup neural network once for the batch with correct scaling
        nn_terminal_cost_norm, build_path = setup_neural_network_for_process(batch_id, JMAX)  # Use JMAX here
        
        for i, data in enumerate(batch_data):
            try:
                x0_val = data['x0_val']
                J_optM = data['J_optM']
                J_optNM = data['J_optNM']
                
                # Define the OCP for the shorter horizon M using terminal cost
                ocp_terminal_cost = OCP_Terminal_Cost(
                    M, 
                    q0=x0_val[:1], 
                    dq0=x0_val[1:], 
                    terminal_cost_func=nn_terminal_cost_norm, 
                    nx=nx, 
                    nu=nj, 
                    w_p=w_p, 
                    w_v=w_v,  
                    w_a=w_a, 
                    dt=dt, 
                    f=f,
                    w_term=w_term_cost,
                )
                
                # Calculate normalized difference (same as single-core code)
                difference_norm = (ocp_terminal_cost - J_optNM) / J_optNM
                
                # Calculate average cost per step
                cmsMs = J_optM / M
                cmsMJs = ocp_terminal_cost / (N+M)  # Note: this should be (N+M) as in single-core
                cmsNMs = J_optNM / (N+M)
                
                result = {
                    'x0_val': x0_val,
                    'ocp_terminal_cost': ocp_terminal_cost,
                    'J_optM': J_optM,
                    'J_optNM': J_optNM,
                    'difference': difference_norm,
                    'cmsMs': cmsMs,
                    'cmsMJs': cmsMJs,
                    'cmsNMs': cmsNMs,
                    'batch_id': batch_id,
                    'item_id': i
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing item {i} in batch {batch_id}: {e}")
                continue
    
    except Exception as e:
        print(f"Error setting up batch {batch_id}: {e}")
        return []
        
    finally:
        # Clean up batch-specific build directory
        if build_path and os.path.exists(build_path):
            try:
                shutil.rmtree(build_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up {build_path}: {cleanup_error}")
    
    return results

def create_batches(items, batch_size):
    """Split items into batches for processing"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def main():
    # Generate all initial states (same as single-core)
    x0_vals = []
    np.random.seed(42)  # For reproducibility
    for _ in range(K):
        x0_val = np.array([np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)])
        x0_vals.append(x0_val)
    
    print(f"Using {ncores} cores for parallel processing")
    print(f"Total number of computations: {K}")
    
    # =============================================================================
    # PHASE 1: Compute J_optM and J_optNM for all states to find JMAX and JMAX_M
    # =============================================================================
    print("\n=== Phase 1: Computing initial costs to find JMAX ===")
    start_time = time.time()
    
    # Prepare arguments for phase 1
    phase1_args = [(x0_val, M, N, nx, nj, w_p, w_v, w_a, dt) for x0_val in x0_vals]
    
    try:
        with mp.Pool(processes=min(ncores, len(x0_vals))) as pool:
            phase1_results = pool.map(compute_initial_costs, phase1_args)
        
        # Filter out None results
        phase1_results = [r for r in phase1_results if r is not None]
        phase1_time = time.time() - start_time
        print(f"Phase 1 completed: {len(phase1_results)}/{K} successful in {phase1_time:.2f}s")
        
        if not phase1_results:
            print("Phase 1 failed completely!")
            return None
        
        # Find JMAX from phase 1 results
        JM_vec = [r['J_optM'] for r in phase1_results]
        JNM_vec = [r['J_optNM'] for r in phase1_results]
        JMAX = np.max(JNM_vec)  # Use max of N+M horizon costs

        print(f"JMAX = {JMAX:.4f}")
        
    except Exception as e:
        print(f"Phase 1 failed: {e}")
        return None
    
    # =============================================================================
    # PHASE 2: Compute terminal costs using the neural network with correct scaling
    # =============================================================================
    print("\n=== Phase 2: Computing terminal costs with neural network ===")
    start_time = time.time()
    
    # Try batch processing first (recommended for L4CasADi)
    batch_size = max(1, len(phase1_results) // ncores)
    batches = list(create_batches(phase1_results, batch_size))
    
    print(f"Processing {len(batches)} batches of size ~{batch_size}")
    
    try:
        # Prepare batch arguments for phase 2
        batch_args = [(batch, JMAX, i) for i, batch in enumerate(batches)]
        
        with mp.Pool(processes=min(ncores, len(batches))) as pool:
            batch_results = pool.map(compute_batch_terminal_cost, batch_args)
        
        # Flatten results from all batches
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        phase2_time = time.time() - start_time
        print(f"Phase 2 completed: {len(results)}/{len(phase1_results)} successful in {phase2_time:.2f}s")
        
    except Exception as e:
        print(f"Batch processing failed, trying individual processes: {e}")
        
        # Fallback to individual processes
        try:
            phase2_args = [(r['x0_val'], r['J_optM'], r['J_optNM'], M, N, nx, nj, w_p, w_v, w_a, dt, w_term_cost, JMAX, i) 
                            for i, r in enumerate(phase1_results)]
            
            with mp.Pool(processes=min(ncores, len(phase1_results))) as pool:
                results = pool.map(compute_terminal_cost, phase2_args)
            
            # Filter out None results
            results = [r for r in results if r is not None]
            phase2_time = time.time() - start_time
            print(f"Individual processing completed: {len(results)}/{len(phase1_results)} successful in {phase2_time:.2f}s")
            
        except Exception as e2:
            print(f"Both batch and individual processing failed: {e2}")
            return None
    
    if not results:
        print("Phase 2 failed completely!")
        return None
    
    print(f"Successfully completed {len(results)} out of {K} computations")
    print(f"Total time: {phase1_time + phase2_time:.2f}s")
    
    # Extract results (same as single-core code)
    differences = [r['difference'] for r in results]
    x0_vals_successful = [r['x0_val'] for r in results]
    cmsM = [r['cmsMs'] for r in results]
    cmsMJ = [r['cmsMJs'] for r in results]
    cmsNM = [r['cmsNMs'] for r in results]
    
    # Plot results (same as single-core code)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(differences)+1), differences, marker='o', linestyle='-', 
             label='(Terminal Cost - Optimal Cost)/Optimal Cost')
    plt.xlabel('Initial State Index (K)')
    plt.ylabel('Cost Difference')
    plt.title('Normalized Cost Difference (M+J) - (N+M)')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cmsM)+1), cmsM, marker='o', linestyle='-', label='cmsM (case 1)')
    plt.plot(range(1, len(cmsMJ)+1), cmsMJ, marker='s', linestyle='--', label='cmsMJ (case 2)')
    plt.plot(range(1, len(cmsNM)+1), cmsNM, marker='^', linestyle='-.', label='cmsNM (case 3)')
    plt.xlabel('Initial State Index (K)')
    plt.ylabel('Cost Values')
    plt.title('Average Cost per Step')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    print(f"Available CPU cores: {mp.cpu_count()}")
    print(f"Using {ncores} cores for parallel processing")
    
    results = main()