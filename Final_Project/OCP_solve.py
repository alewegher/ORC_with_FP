import casadi as cs
import numpy as np


def OCP_cost(N,q0,dq0,nx,nu,w_p,w_v,w_a,dt,f):
    
    """
    Function to solve the Optimal Control Problem (OCP) using CasADi.
    in this case we have a incomplete Bolza's form OCP, we have to optimize only the effect of the Lagrange-running cost function.
    """
    
    optimizer = cs.Opti()
    # define the containers for the state and control trajectories
    X,U = [], []
    # Initialize the state trajectory with the initial state
    for i in range(N):
        X.append(optimizer.variable(nx)) # state vector at each time step
    for i in range(N-1):
        U.append(optimizer.variable(nu)) # controls vector at each time step
        
    cost = 0.
    
    # update cost function
    
    for i in range(N-1):
        cost += w_p * cs.sumsqr(X[i][:nx]) # position cost
        cost += w_v * cs.sumsqr(X[i][nx:]) # velocity cost
        cost += w_a * cs.sumsqr(U[i])      # acceleration cost
        optimizer.subject_to(X[i+1] == X[i] + dt * f(cs.vec(X[i]), cs.vec(U[i]))) # dynamics constraint
        
    # set the constraints for the optimazion problem
    
    # initial state
    #optimizer.subject_to(X[0] == cs.vertcat(q0,dq0)) # initial state constraint
    
    optimizer.subject_to(X[0]== cs.vertcat(*q0,*dq0))
    
    optimizer.minimize(cost) # set the cost function to minimize
    
    # set the solver options
    
    opts = {
     "ipopt.print_level": 0,
     "ipopt.tol": 1e-6,
     "ipopt.compl_inf_tol": 1e-6,
     "print_time": 0,                # print information about execution time
     "detect_simple_bounds": True,
     "ipopt.hessian_approximation": "limited-memory", # Hessian approximation
    }
    
    optimizer.solver("ipopt", opts)
    
    solution = optimizer.solve() # solve the optimization problem
    
    Optimal_Cost = solution.value(cost) # get the optimal cost
  
    return Optimal_Cost # return the state and control trajectories, and the optimal cost


def OCP_Terminal_Cost(N,q0,dq0,nx,nu,w_p,w_v,w_a,dt,f,w_term,terminal_cost_func): # here we have complete OCP in Bolza's form
    """
    Function to solve the Optimal Control Problem (OCP) using CasADi.
    in this case we have a complete Bolza's form OCP, we have to optimize both the effect of the Lagrange-running cost function and the cost associated to the desired final state.
    """
    
    optimizer = cs.Opti()
    init_x = optimizer.variable(nx)
    
    X,U = [], []
    
    for i in range(N):
        X.append(optimizer.variable(nx))
    
    for i in range(N-1):    
        U.append(optimizer.variable(nu))
    
    
    cost = 0.
    
    for i in range(N-1):
        cost += w_p * cs.sumsqr(X[i][:nx])
        cost += w_v * cs.sumsqr(X[i][nx:])
        cost += w_a * cs.sumsqr(U[i])
        optimizer.subject_to(X[i+1] == X[i] + dt * f(X[i], U[i]))
        
    # set the constraints for the optimazion problem
    optimizer.subject_to(X[0] == cs.vertcat(q0,dq0)) # initial state constraint
    cost += w_term*terminal_cost_func(X[N-1]) # add the terminal cost function to the cost function
    optimizer.minimize(cost) # set the cost function to minimize 
    
    # set the solver options
    opts = {
     "ipopt.print_level": 0,
     "ipopt.tol": 1e-4,
     "ipopt.compl_inf_tol": 1e-4,
     "print_time": 0,                # print information about execution time
     "detect_simple_bounds": True,
     "ipopt.hessian_approximation": "limited-memory", #Hessian approximation
    }
    optimizer.solver("ipopt", opts)

    solution = optimizer.solve() # solve the optimization problem
    Optimal_Cost = solution.value(cost) # get the optimal cost
    
    return Optimal_Cost # return the state and control trajectories, and the optimal cost



def test_ocp():
    # Parametri di esempio
    N = 20
    nq = 2
    nx = 2 * nq
    nu = nq
    q0 = np.ones(nq)
    dq0 = np.ones(nq)
    w_p = 1.0
    w_v = 0.1
    w_a = 0.01
    dt = 0.1

    # Dinamica fittizia: x_dot = Ax + Bu
    def f(x, u):
        A = 0.8*np.eye(nx)
        B = np.eye(nx, nu)
        return A @ x + B @ u

    # Terminal cost fittizio
    def terminal_cost(x):
        return cs.sumsqr(x)

    # Test OCP_cost
    cost_opt = OCP_cost(N, q0, dq0, nx, nu, w_p, w_v, w_a, dt, f)
    print("OCP_cost: optimal cost =", cost_opt)

    # Test OCP_Terminal_Cost
    cost_terminal = OCP_Terminal_Cost(N, q0, dq0, nx, nu, w_p, w_v, w_a, dt, f, terminal_cost)
    print("OCP_Terminal_Cost: optimal cost =", cost_terminal)
    
    
    
if __name__ == "__main__":
    test_ocp()
# This code is a test function to verify the OCP_cost and OCP_Terminal_Cost functions.
    
     