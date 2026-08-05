import os
import numpy as np
import casadi as cs
from casadi import Function, MX
import torch
from l4casadi import L4CasADi as NNC
from orc.Final_Project.OCP_solve import OCP_Terminal_Cost, OCP_cost
from orc.Final_Project.Double_Pend.Double_pend_training import Pendulum, ExtendedNeuralNetwork
from example_robot_data.robots_loader import load
import matplotlib.pyplot as plt
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import pickle
import tempfile
import shutil

#! As for the Single Pendulum case I made AI convert the code I wrote for single core usage into multi-core usage to generate faster the results.!#

def get_available_ram_gb():
    """Read MemAvailable from /proc/meminfo (Linux). Falls back to a conservative 4 GB if unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)  # kB -> GB
    except OSError:
        pass
    return 4.0

def get_safe_worker_count(mem_per_worker_gb=1.5, reserve_gb=4.0):
    """
    Cap the number of parallel worker processes so we don't exceed available RAM.
    Each worker (torch + L4CasADi build + ipopt) costs roughly mem_per_worker_gb;
    reserve_gb is left untouched for the OS/other processes.
    """
    cpu_cap = max(1, multiprocessing.cpu_count() - 4)
    available_gb = get_available_ram_gb()
    ram_cap = max(1, int((available_gb - reserve_gb) // mem_per_worker_gb))
    return max(1, min(cpu_cap, ram_cap))

# Strategy 1: Process-based parallelization with model recreation
def create_nn_model_and_casadi_func(device_str="cpu"):
    """Create neural network model and CasADi function in each process"""
    device = torch.device(device_str)
    
    # Define the neural network model
    nx = 4  # Will be passed as parameter
    HiddenLayers = 128
    OutputLayers = 1
    activation = torch.nn.Tanh()
    ModelNN = ExtendedNeuralNetwork(nx, HiddenLayers, OutputLayers, activation, ub=1)
    
    # Load checkpoint
    checkpoint = torch.load("/home/student/shared/orc/Final_Project/Double_Pend/nn_cost_pred.pt", 
                           map_location=device)
    ModelNN.load_state_dict(checkpoint["model_state_dict"])
    ModelNN.eval()
    ModelNN.to(device)
    
    # Create unique build directory for this process
    process_id = os.getpid()
    build_path = os.path.expanduser(f"/tmp/l4c_gen_process_{process_id}")
    os.makedirs(build_path, exist_ok=True)
    
    # Convert to CasADi function
    l4c_model = NNC(ModelNN, device=device_str, build_dir=build_path)
    x_sym = MX.sym("x", 1, nx)
    nn_terminal_cost = Function("nn_terminal_cost", [x_sym], [l4c_model(x_sym)])
    
    return nn_terminal_cost, l4c_model, build_path

def compute_single_cost(args):
    """Compute cost for a single initial state - runs in separate process"""
    x0_val, M, N, nx, nq, nu, w_p, w_v, w_a, w_term_cost, dt, params_pend, device_str = args
    
    try:
        # Create pendulum dynamics
        d_pend = Pendulum(params_pend)
        f = d_pend.dynamics(cs.MX.sym('q', nq), cs.MX.sym('dq', nq), cs.MX.sym('u', nu))
        
        # First compute the costs without terminal cost
        J_optM = OCP_cost(M, x0_val[:nq], x0_val[nq:], nx, nq, w_p, w_v, w_a, dt, f)
        J_optNM = OCP_cost(M + N, x0_val[:nq], x0_val[nq:], nx, nq, w_p, w_v, w_a, dt, f)

        # Calculate average costs per step
        cmsMs = J_optM / M
        cmsNMs = J_optNM / (N+M)
        
        return {
            'x0_val': x0_val,
            'J_optM': J_optM,
            'J_optNM': J_optNM,
            'cmsMs': cmsMs,
            'cmsNMs': cmsNMs
        }
        
    except Exception as e:
        print(f"Error in process {os.getpid()}: {e}")
        return None

def compute_terminal_cost(args):
    """Compute terminal cost with neural network prediction"""
    x0_val, M, N, nx, nq, nu, w_p, w_v, w_a, w_term_cost, dt, params_pend, device_str, JMAX = args
    
    try:
        # Create pendulum dynamics and NN model
        d_pend = Pendulum(params_pend)
        f = d_pend.dynamics(cs.MX.sym('q', nq), cs.MX.sym('dq', nq), cs.MX.sym('u', nu))
        nn_terminal_cost, l4c_model, build_path = create_nn_model_and_casadi_func(device_str)
        
        # Update the neural network's output scaling with JMAX
        x_sym = MX.sym("x", 1, nx)
        nn_terminal_cost = Function("nn_terminal_cost", [x_sym], [JMAX * (l4c_model(x_sym))])
        
        # Compute terminal cost OCP
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
        
        # Calculate average cost per step
        cmsMJs = ocp_terminal_cost / (N+M)
        
        # Cleanup build directory
        if os.path.exists(build_path):
            shutil.rmtree(build_path, ignore_errors=True)
        
        return {
            'x0_val': x0_val,
            'cmsMJs': cmsMJs,
            'ocp_terminal_cost': ocp_terminal_cost
        }
        
    except Exception as e:
        print(f"Error in process {os.getpid()}: {e}")
        return None

# Strategy 3: Memory-mapped shared arrays for large-scale computation
def create_shared_memory_computation():
    """Create computation using shared memory for very large problems"""
    import multiprocessing.shared_memory as sm
    
    def worker_with_shared_memory(shared_name, shape, dtype, start_idx, end_idx, args):
        """Worker that uses shared memory for results"""
        # Attach to existing shared memory
        existing_shm = sm.SharedMemory(name=shared_name)
        shared_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        M, N, nx, nq, nu, w_p, w_v, w_a, w_term_cost, dt, params_pend, device_str, x0_vals = args
        
        # Create model for this worker
        nn_terminal_cost, l4c_model, build_path = create_nn_model_and_casadi_func(device_str)
        d_pend = Pendulum(params_pend)
        f = d_pend.dynamics(cs.MX.sym('q', nq), cs.MX.sym('dq', nq), cs.MX.sym('u', nu))
        
        try:
            for i in range(start_idx, end_idx):
                if i >= len(x0_vals):
                    break
                    
                x0_val = x0_vals[i]
                
                # Compute costs
                ocp_terminal_cost = OCP_Terminal_Cost(
                    M, q0=x0_val[:nq], dq0=x0_val[nq:], 
                    terminal_cost_func=nn_terminal_cost, nx=nx, nu=nu,
                    w_p=w_p, w_v=w_v, w_term=w_term_cost, w_a=w_a, dt=dt, f=f
                )
                
                J_optM = OCP_cost(M, x0_val[:nq], x0_val[nq:], nx, nq, w_p, w_v, w_a, dt, f)
                J_optNM = OCP_cost(M + N, x0_val[:nq], x0_val[nq:], nx, nq, w_p, w_v, w_a, dt, f)
                
                # Store results in shared memory
                shared_array[i, 0] = np.abs(ocp_terminal_cost - J_optNM)  # difference
                shared_array[i, 1] = J_optM / M  # cmsMs
                shared_array[i, 2] = ocp_terminal_cost / (M + N)  # cmsMJs
                shared_array[i, 3] = J_optNM / (M + N)  # cmsNMs
                
        finally:
            # Cleanup
            existing_shm.close()
            if os.path.exists(build_path):
                shutil.rmtree(build_path, ignore_errors=True)
    
    return worker_with_shared_memory

# Strategy 2: Batch processing with shared model
def compute_batch_costs(x0_batch, M, N, nx, nq, nu, w_p, w_v, w_a, w_term_cost, dt, params_pend, device_str="cpu"):
    """Compute costs for a batch of initial states"""

    # Create pendulum dynamics once
    d_pend = Pendulum(params_pend)
    f = d_pend.dynamics(cs.MX.sym('q', nq), cs.MX.sym('dq', nq), cs.MX.sym('u', nu))

    # Create neural network model once for the batch
    nn_terminal_cost, l4c_model, build_path = create_nn_model_and_casadi_func(device_str)

    results = []

    for x0_val in x0_batch:
        try:
            # Compute costs
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

            J_optM = OCP_cost(M, x0_val[:nq], x0_val[nq:], nx, nq, w_p, w_v, w_a, dt, f)
            J_optNM = OCP_cost(M + N, x0_val[:nq], x0_val[nq:], nx, nq, w_p, w_v, w_a, dt, f)

            # Calculate metrics
            difference = np.abs(ocp_terminal_cost - J_optNM)
            cmsMs = J_optM / M
            cmsMJs = ocp_terminal_cost / (M + N)
            cmsNMs = J_optNM / (M + N)

            results.append({
                'x0_val': x0_val,
                'difference': difference,
                'cmsMs': cmsMs,
                'cmsMJs': cmsMJs,
                'cmsNMs': cmsNMs,
                'ocp_terminal_cost': ocp_terminal_cost,
                'J_optM': J_optM,
                'J_optNM': J_optNM
            })

        except Exception as e:
            print(f"Error processing state {x0_val}: {e}")
            results.append(None)

    # Cleanup build directory
    if os.path.exists(build_path):
        shutil.rmtree(build_path, ignore_errors=True)

    return results

def main_parallel_computation():
    """Main function demonstrating parallel computation strategies"""
    
    # Load the double pendulum robot model
    robot = load("double_pendulum")
    
    # Model parameters
    nq = robot.model.nq
    nv = robot.model.nv
    nu = 2
    nx = 2 * nq
    
    # Simulation parameters
    dt = 0.01
    N = 200
    K = 50
    M = math.ceil(0.35 * N)
    
    # Cost weights
    w_p = 1e-1
    w_v = 1e-2
    w_a = 1e-2
    w_term_cost = 1.0
    
    # Pendulum parameters
    params_pend = {
        'nq': nq,
        'L1': robot.model.jointPlacements[1].translation[2],
        'L2': robot.model.jointPlacements[2].translation[2],
        'm1': robot.model.inertias[1].mass,
        'm2': robot.model.inertias[2].mass,
        'g': 9.81,
        'J1': robot.model.inertias[1].inertia[0, 0],
        'J2': robot.model.inertias[2].inertia[0, 0],
        'b1': 0.05,
        'b2': 0.05,
        'COM1': robot.model.inertias[1].lever[2],
        'COM2': robot.model.inertias[2].lever[2]
    }
    
    # Generate initial states
    x0_vals = []
    for _ in range(K):
        x0_val = np.concatenate([
            np.random.uniform(low=-np.pi, high=np.pi, size=nq),
            np.random.uniform(low=-2*np.pi, high=2*np.pi, size=nq)
        ])
        x0_vals.append(x0_val)
    
    # GPU is reserved for training / a single one-off inference; spawning it into every
    # worker process here would have each grab its own CUDA context and L4CasADi build,
    # which blows past the GPU's VRAM (and drags down host RAM) once more than a few
    # processes are alive at once. The ipopt solve dominates runtime anyway, so the NN
    # forward pass being on CPU costs nothing noticeable.
    device_str = "cpu"
    n_cores = get_safe_worker_count()

    print(f"Using {n_cores} cores for parallel computation (RAM-capped, available: {get_available_ram_gb():.1f} GB)")
    print(f"Computing costs for {K} initial states")
    
    # Prepare arguments for parallel processing
    args_list = [
        (x0_val, M, N, nx, nq, nu, w_p, w_v, w_a, w_term_cost, dt, params_pend, device_str)
        for x0_val in x0_vals
    ]

    # Strategy 2: Batch processing (alternative approach)
    print("\n=== Strategy 2: Batch processing ===")
    
    batch_size = max(1, K // n_cores)
    batches = [x0_vals[i:i + batch_size] for i in range(0, len(x0_vals), batch_size)]

    batch_results = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        batch_args = [
            (batch, M, N, nx, nq, nu, w_p, w_v, w_a, w_term_cost, dt, params_pend, device_str)
            for batch in batches
        ]
        
        futures = [executor.submit(compute_batch_costs, *args) for args in batch_args]
        
        for future in as_completed(futures):
            try:
                batch_result = future.result()
                batch_results.extend(batch_result)
                print(f"Completed batch, total results: {len(batch_results)}")
            except Exception as e:
                print(f"Error in batch: {e}")
    
    # First phase: Compute costs without terminal cost
    results = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_idx = {executor.submit(compute_single_cost, args): i 
                        for i, args in enumerate(args_list)}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result is not None:
                    results.append((idx, result))
                    print(f"Completed computation {len(results)}/{K}")
            except Exception as e:
                print(f"Error in future {idx}: {e}")
    
    # Sort results and compute JMAX
    results.sort(key=lambda x: x[0])
    cmsNM = [r[1]['cmsNMs'] for r in results if r[1] is not None]
    JMAX = np.max(cmsNM)
    
    # Second phase: Compute terminal costs with updated JMAX
    terminal_args_list = [
        (*args, JMAX) for args in args_list
    ]
    
    terminal_results = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_idx = {executor.submit(compute_terminal_cost, args): i 
                        for i, args in enumerate(terminal_args_list)}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result is not None:
                    terminal_results.append((idx, result))
            except Exception as e:
                print(f"Error in future {idx}: {e}")
    
    # Sort terminal results
    terminal_results.sort(key=lambda x: x[0])
    
    # Process all results for plotting
    if results and terminal_results:
        # --- Simulate & plot trajectories for one example initial state (the 3 cases) ---
        # Done directly in the main process (not via the pool) since it's a single solve, not K of them.
        x0_example = x0_vals[0]
        d_pend_ex = Pendulum(params_pend)
        f_ex = d_pend_ex.dynamics(cs.MX.sym('q', nq), cs.MX.sym('dq', nq), cs.MX.sym('u', nu))
        nn_terminal_cost_ex, l4c_model_ex, build_path_ex = create_nn_model_and_casadi_func(device_str)
        x_sym_ex = MX.sym("x", 1, nx)
        nn_terminal_cost_ex = Function("nn_terminal_cost_ex", [x_sym_ex], [JMAX * (l4c_model_ex(x_sym_ex))])

        J_M_ex,  X_M_traj,  U_M_traj  = OCP_cost(M, x0_example[:nq], x0_example[nq:], nx, nq, w_p, w_v, w_a, dt, f_ex, return_traj=True)
        J_NM_ex, X_NM_traj, U_NM_traj = OCP_cost(M + N, x0_example[:nq], x0_example[nq:], nx, nq, w_p, w_v, w_a, dt, f_ex, return_traj=True)
        J_MJ_ex, X_MJ_traj, U_MJ_traj = OCP_Terminal_Cost(
            N,
            q0=x0_example[:nq],
            dq0=x0_example[nq:],
            terminal_cost_func=nn_terminal_cost_ex,
            nx=nx,
            nu=nu,
            w_p=w_p,
            w_v=w_v,
            w_term=w_term_cost,
            w_a=w_a,
            dt=dt,
            f=f_ex,
            return_traj=True
        )

        if os.path.exists(build_path_ex):
            shutil.rmtree(build_path_ex, ignore_errors=True)

        t_M  = np.arange(X_M_traj.shape[0]) * dt
        t_NM = np.arange(X_NM_traj.shape[0]) * dt
        t_MJ = np.arange(X_MJ_traj.shape[0]) * dt

        traj_cases = [
            ('case 1: M (no terminal)',   t_M,  X_M_traj,  U_M_traj),
            ('case 2: M+J (NN terminal)', t_MJ, X_MJ_traj, U_MJ_traj),
            ('case 3: M+N (no terminal)', t_NM, X_NM_traj, U_NM_traj),
        ]

        fig, axs = plt.subplots(3, nq, figsize=(12, 9), sharex='col')
        for j in range(nq):
            for label, t, X_traj, U_traj in traj_cases:
                axs[0, j].plot(t, X_traj[:, j], label=label)
                axs[1, j].plot(t, X_traj[:, nq + j], label=label)
                axs[2, j].plot(t[:-1], U_traj[:, j], label=label)
            axs[0, j].set_title(f'q{j+1} (position)')
            axs[1, j].set_title(f'dq{j+1} (velocity)')
            axs[2, j].set_title(f'u{j+1} (torque)')
            axs[2, j].set_xlabel('time [s]')
            axs[0, j].grid()
            axs[1, j].grid()
            axs[2, j].grid()
        axs[0, 0].legend()
        fig.suptitle(f'Simulated trajectories, x0 = {x0_example}')
        plt.tight_layout()

        differences = []
        cmsM = []
        cmsMJ = []
        cmsNM = []
        
        for i in range(len(results)):
            if results[i][1] and terminal_results[i][1]:
                r1 = results[i][1]
                r2 = terminal_results[i][1]
                
                # Calculate normalized difference as per single core version
                diff = (r2['cmsMJs'] - r1['cmsNMs']) / r1['cmsNMs']
                differences.append(diff)
                cmsM.append(r1['cmsMs'])
                cmsMJ.append(r2['cmsMJs'])
                cmsNM.append(r1['cmsNMs'])
        
        # Normalize differences
        JMAX = np.max(np.array(cmsM + cmsMJ + cmsNM))
        differences = np.array(differences) / JMAX
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(differences) + 1), differences, marker='o', linestyle='-', 
                label='|Terminal Cost - Optimal Cost|')
        plt.xlabel('Initial State Index')
        plt.ylabel('Normalized Cost Difference')
        plt.title('Cost Difference (M+J) - (N+M)')
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cmsM) + 1), cmsM, marker='o', linestyle='-', label='cmsM (case 1)')
        plt.plot(range(1, len(cmsMJ) + 1), cmsMJ, marker='s', linestyle='--', label='cmsMJ (case 2)')
        plt.plot(range(1, len(cmsNM) + 1), cmsNM, marker='^', linestyle='-.', label='cmsNM (case 3)')
        plt.xlabel('Initial State Index')
        plt.ylabel('Cost Values')
        plt.title('Average Cost per Step')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Computation completed successfully!")
        print(f"Mean cost difference: {np.mean(differences):.6f}")
        print(f"Max cost difference: {np.max(differences):.6f}")
        print(f"Std of cost differences: {np.std(differences):.6f}")
        
        
if __name__ == "__main__":
    # CUDA contexts can't survive fork(); child processes must start clean.
    multiprocessing.set_start_method("spawn", force=True)
    main_parallel_computation()