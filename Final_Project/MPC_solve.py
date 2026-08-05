import os
import time
import numpy as np
import casadi as cs
from orc.Final_Project.OCP_solve import OCP_cost

"""
Closed-loop (receding-horizon) MPC simulation, shared between the single and
double pendulum projects. Unlike OCP_solve.py (single-shot open-loop cost
comparison), this module re-solves an OCP every plant timestep, applies only
the first control, and forward-integrates the REAL dynamics with it.

The running cost here uses a correct q/dq split (X[i][:nq], X[i][nq:2*nq]).
This deliberately does NOT reuse OCP_cost/OCP_Terminal_Cost's own running-cost
construction, which indexes X[i][:nx]/X[i][nx:] where X[i] already has length
nx -- i.e. the "position" slice is the whole state and the "velocity" slice is
always empty, so w_v has no effect there. That behavior is left untouched in
OCP_solve.py (the open-loop *_comp.py comparisons depend on it); it's only
fixed here, where a physically meaningful cost is required for the controller
to actually behave sensibly in closed loop.
"""

DEFAULT_SOLVER_OPTS = {
    "ipopt.print_level": 0,
    "ipopt.tol": 1e-4,
    "ipopt.compl_inf_tol": 1e-6,
    "print_time": 0,
    "detect_simple_bounds": True,
    "ipopt.hessian_approximation": "limited-memory",
}


def build_mpc_ocp(N, nq, nx, nu, dt, f, w_p, w_v, w_a, w_term=0.0, terminal_cost_func=None, solver_opts=None):
    """
    Build one Opti-based receding-horizon OCP. Meant to be solved repeatedly
    by run_mpc_closed_loop -- never rebuilt per simulation step.

    N follows OCP_solve.py's convention: X has length N (X[0..N-1]), U has
    length N-1 (U[0..N-2]). Passing M or M+N straight from the *_comp.py
    scripts gives exact horizon parity with the existing open-loop comparison.

    Returns (opti, X, U, param_x_init).
    """
    opti = cs.Opti()

    X = [opti.variable(nx) for _ in range(N)]
    U = [opti.variable(nu) for _ in range(N - 1)]
    param_x_init = opti.parameter(nx)

    cost = 0.0
    for i in range(N - 1):
        cost += w_p * cs.sumsqr(X[i][:nq])
        cost += w_v * cs.sumsqr(X[i][nq:2 * nq])
        cost += w_a * cs.sumsqr(U[i])
        opti.subject_to(X[i + 1] == X[i] + dt * f(X[i], U[i]))

    opti.subject_to(X[0] == param_x_init)

    if terminal_cost_func is not None:
        cost += w_term * terminal_cost_func(X[N - 1])

    opti.minimize(cost)

    opts = dict(solver_opts or DEFAULT_SOLVER_OPTS)
    opti.solver("ipopt", opts)

    return opti, X, U, param_x_init


def run_mpc_closed_loop(opti, X, U, param_x_init, f, x0, T_sim, dt,
                         warm_start=True, max_iter_first=1000, max_iter_receding=50,
                         solver_opts=None, verbose=True):
    """
    Receding-horizon closed-loop simulation. Reuses the SAME opti/X/U for
    every step -- required for warm-starting.

    Each step: shift the previous solution into the initial guess (duplicate
    the last entry), solve, take u0 = U[0], integrate the REAL plant one step
    with explicit Euler using the same f(state,u)->xdot used everywhere in
    this repo, log the result, repeat.

    Returns X_log (n_steps+1, nx), U_log (n_steps, nu), t_log (n_steps+1,).
    """
    N = len(X)
    nx = X[0].shape[0]
    nu = U[0].shape[0]
    n_steps = int(round(T_sim / dt))

    X_log = np.zeros((n_steps + 1, nx))
    U_log = np.zeros((n_steps, nu))
    t_log = np.arange(n_steps + 1) * dt

    x = np.asarray(x0, dtype=float).flatten()
    X_log[0] = x

    base_opts = dict(solver_opts or DEFAULT_SOLVER_OPTS)

    # First solve: full convergence, no warm start available yet.
    first_opts = dict(base_opts)
    first_opts["ipopt.max_iter"] = max_iter_first
    opti.solver("ipopt", first_opts)
    opti.set_value(param_x_init, x)
    sol = opti.solve()

    # Subsequent solves: capped iterations, relying on warm start to stay close to optimal.
    if max_iter_receding is not None:
        receding_opts = dict(base_opts)
        receding_opts["ipopt.max_iter"] = max_iter_receding
        opti.solver("ipopt", receding_opts)

    for k in range(n_steps):
        if warm_start:
            for t in range(N - 1):
                opti.set_initial(X[t], sol.value(X[t + 1]))
            opti.set_initial(X[N - 1], sol.value(X[N - 1]))
            for t in range(N - 2):
                opti.set_initial(U[t], sol.value(U[t + 1]))
            if N - 2 >= 0:
                opti.set_initial(U[N - 2], sol.value(U[N - 2]))
            opti.set_initial(opti.lam_g, sol.value(opti.lam_g))

        opti.set_value(param_x_init, x)
        try:
            sol = opti.solve()
        except RuntimeError:
            sol = opti.debug

        u0 = np.asarray(sol.value(U[0])).flatten()
        xdot = np.asarray(f(cs.DM(x), cs.DM(u0)).full()).flatten()
        x = x + dt * xdot

        X_log[k + 1] = x
        U_log[k] = u0

        if verbose and (k % 50 == 0 or k == n_steps - 1):
            stats = sol.stats()
            print(f"  step {k + 1}/{n_steps}  |x|={np.linalg.norm(x):.4f}  "
                  f"iters={stats.get('iter_count')}  status={stats.get('return_status')}")

    return X_log, U_log, t_log


def compute_jmax(K, nq, nx, w_p, w_v, w_a, dt, f, cost_horizon):
    """
    Calibrate the NN terminal-cost scale the same way the *_comp.py scripts
    do: draw K random states, compute the average cost per step over
    `cost_horizon` steps with OCP_solve.OCP_cost (its own -- unfixed --
    running cost, since that's the same scale the terminal-cost NN was
    trained against), and take the max.
    """
    cmsNM = []
    for _ in range(K):
        q0 = np.random.uniform(low=-np.pi, high=np.pi, size=nq)
        dq0 = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=nq)
        J = OCP_cost(cost_horizon, q0, dq0, nx, nq, w_p, w_v, w_a, dt, f)
        cmsNM.append(J / cost_horizon)
    return float(np.max(cmsNM))


def solve_mpc_case(label, N, nq, nx, nu, dt, f, w_p, w_v, w_a, x0, T_sim,
                    w_term=0.0, terminal_cost_func=None,
                    warm_start=True, max_iter_first=1000, max_iter_receding=50,
                    solver_opts=None, verbose=True):
    """Convenience wrapper: build_mpc_ocp(...) + run_mpc_closed_loop(...)."""
    if verbose:
        print(f"\n=== {label} (N_pred={N}) ===")

    opti, X, U, param_x_init = build_mpc_ocp(
        N, nq, nx, nu, dt, f, w_p, w_v, w_a,
        w_term=w_term, terminal_cost_func=terminal_cost_func, solver_opts=solver_opts
    )

    t_start = time.time()
    X_log, U_log, t_log = run_mpc_closed_loop(
        opti, X, U, param_x_init, f, x0, T_sim, dt,
        warm_start=warm_start, max_iter_first=max_iter_first, max_iter_receding=max_iter_receding,
        solver_opts=solver_opts, verbose=verbose
    )

    if verbose:
        print(f"  {label}: done in {time.time() - t_start:.1f}s")

    return {'label': label, 't': t_log, 'X': X_log, 'U': U_log}


def _finalize_and_maybe_save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_mpc_kinematics(cases, nq, suptitle=None, save_path=None):
    """
    2-row (q / dq) x nq-column subplot grid, overlaying every case per joint.
    squeeze=False is required so nq=1 (single pendulum) still yields a 2-D
    axs array, matching the axs[row, col] indexing used below. If save_path
    is given, the figure is also saved there (parent dirs created as needed).
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, nq, figsize=(6 * nq, 6), squeeze=False)

    for j in range(nq):
        for case in cases:
            t, X_log = case['t'], case['X']
            axs[0, j].plot(t, X_log[:, j], label=case['label'])
            axs[1, j].plot(t, X_log[:, nq + j], label=case['label'])

        axs[0, j].set_title(f'q{j + 1} (position)')
        axs[1, j].set_title(f'dq{j + 1} (velocity)')
        axs[0, j].set_ylabel('q [rad]')
        axs[1, j].set_ylabel('dq [rad/s]')
        axs[1, j].set_xlabel('time [s]')
        for row in range(2):
            axs[row, j].grid()

    axs[0, 0].legend()
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return _finalize_and_maybe_save(fig, save_path)


def plot_mpc_torque(cases, nq, suptitle=None, save_path=None):
    """1-row x nq-column subplot grid of applied torques u_j(t)."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, nq, figsize=(6 * nq, 3.5), squeeze=False)

    for j in range(nq):
        for case in cases:
            t, U_log = case['t'], case['U']
            axs[0, j].plot(t[:-1], U_log[:, j], label=case['label'])

        axs[0, j].set_title(f'u{j + 1} (torque)')
        axs[0, j].set_xlabel('time [s]')
        axs[0, j].set_ylabel('u [Nm]')
        axs[0, j].grid()

    axs[0, 0].legend()
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return _finalize_and_maybe_save(fig, save_path)
