import os
import numpy as np
import casadi as cs
from casadi import Function, MX
import torch
from l4casadi import L4CasADi as NNC
from orc.Final_Project.MPC_solve import compute_jmax, solve_mpc_case, plot_mpc_kinematics, plot_mpc_torque
from orc.Final_Project.Double_Pend.Double_pend_training import Pendulum, ExtendedNeuralNetwork
from example_robot_data.robots_loader import load
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

"""
Closed-loop MPC simulation for the double pendulum, comparing the same 3
formulations Double_pend_comp.py compares open-loop (cost only): short
horizon M with no terminal cost, long horizon M+N with no terminal cost, and
short horizon M with a learned NN terminal cost. Here each formulation
actually drives the pendulum in receding-horizon fashion for T_sim seconds,
from the same initial state, and the resulting q/dq/u trajectories are
plotted for comparison.
"""

# Load the double pendulum robot model (same setup as Double_pend_comp.py)
robot = load("double_pendulum")

nq = robot.model.nq
nu = 2
nx = 2 * nq

# Simulation parameters
dt = 0.01
N = 200
M = math.ceil(0.35 * N)
K = 50            # random states used to calibrate JMAX (NN terminal cost scale)
T_sim = 5.0       # seconds of closed-loop simulation, same for all 3 cases

# Cost weights
w_p = 1e-1
w_v = 1e-2
w_a = 1e-2
w_term_cost = 1.0

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

d_pend = Pendulum(params_pend)
f = d_pend.dynamics(cs.MX.sym('q', nq), cs.MX.sym('dq', nq), cs.MX.sym('u', nu))

# Load the trained neural network model (same setup as Double_pend_comp.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HiddenLayers = 128
OutputLayers = 1
activation = torch.nn.Tanh()
ModelNN = ExtendedNeuralNetwork(nx, HiddenLayers, OutputLayers, activation, ub=1)
checkpoint = torch.load("/home/student/shared/orc/Final_Project/Double_Pend/nn_cost_pred.pt", map_location=device)
ModelNN.load_state_dict(checkpoint["model_state_dict"])
ModelNN.eval()
ModelNN.to(device)

build_path = os.path.expanduser("/home/student/shared/orc/Final_Project/Double_Pend/_l4c_gen")
os.makedirs(build_path, exist_ok=True)

l4c_model = NNC(ModelNN, device=str(device), build_dir=build_path)
x_sym = MX.sym("x", 1, nx)


if __name__ == "__main__":
    # Fixed, seeded initial state -- reused across all 3 cases (fair comparison)
    # and reproducible across runs.
    np.random.seed(0)
    x0 = np.concatenate([
        np.random.uniform(low=-np.pi, high=np.pi, size=nq),
        np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=nq)
    ])
    print(f"x0 = {x0}")

    # Calibrate the NN terminal-cost scale the same way Double_pend_comp.py does:
    # JMAX = max average cost-per-step over K random states, horizon M+N.
    print(f"Calibrating JMAX over {K} random states...")
    JMAX = compute_jmax(K, nq, nx, w_p, w_v, w_a, dt, f, cost_horizon=M + N)
    print(f"JMAX = {JMAX:.6f}")
    nn_terminal_cost = Function("nn_terminal_cost", [x_sym], [JMAX * (l4c_model(x_sym))])

    cases = [
        solve_mpc_case(
            "case 1: M (no terminal)", N=M, nq=nq, nx=nx, nu=nu, dt=dt, f=f,
            w_p=w_p, w_v=w_v, w_a=w_a, x0=x0, T_sim=T_sim
        ),
        solve_mpc_case(
            "case 2: M+N (no terminal)", N=M + N, nq=nq, nx=nx, nu=nu, dt=dt, f=f,
            w_p=w_p, w_v=w_v, w_a=w_a, x0=x0, T_sim=T_sim
        ),
        solve_mpc_case(
            "case 3: M + NN terminal", N=M, nq=nq, nx=nx, nu=nu, dt=dt, f=f,
            w_p=w_p, w_v=w_v, w_a=w_a, x0=x0, T_sim=T_sim,
            w_term=w_term_cost, terminal_cost_func=nn_terminal_cost
        ),
    ]

    if os.path.exists(build_path):
        import shutil
        shutil.rmtree(build_path, ignore_errors=True)

    FIGURES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "saved_images"))
    os.makedirs(FIGURES_DIR, exist_ok=True)
    suptitle = f"Double pendulum closed-loop MPC, x0 = {x0}"
    plot_mpc_kinematics(cases, nq, suptitle=suptitle, save_path=os.path.join(FIGURES_DIR, "DP_MPC_traj_kin.png"))
    plot_mpc_torque(cases, nq, suptitle=suptitle, save_path=os.path.join(FIGURES_DIR, "DP_MPC_traj_trq.png"))

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
