import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("oracle_2025A1"))
from oracle_2025A1 import oq1,oq2f,oq2g,oq3 


SR_No = 24528 

Q_a,Q_b,Q_c,Q_d,Q_e = oq1(24528)
f = oq2f(24528,np.array([1,2,3,4,5]))
g = oq2g(24528,np.array([1,2,3,4,5]))

# A,b = oq3(24528)
# print(A.shape)
# print(b.shape)


# QUESTION 1

#  1. Minimize f (x) using exact line search. Report x* and f (x*) for each Q. 
def exact_line_search(Q, b, x0, iter=100, tol=1e-10):
    x = x0
    x_star = -np.linalg.inv(Q) @ b
    errors = []

    for i in range(iter):
        grad_fx = (Q @ x) + b
        alpha = (np.linalg.norm(grad_fx) ** 2) /  (grad_fx.T @ Q @ grad_fx)

        errors.append(np.linalg.norm(x-x_star))
        x = x - alpha * grad_fx

        if np.linalg.norm(grad_fx)<tol:
            break
    
    return x, 0.5 * (x.T @ Q @ x) + (b.T @ x), x_star, errors   # x_minimum, f(x_minimum), x_analytical_solution, errors

def plot_errors(errors, title=None, savepath=None):
    iterations = range(len(errors))
    plt.figure(figsize=(6,4))
    plt.plot(iterations, errors, marker="o")
    plt.xlabel("Iteration k")
    plt.ylabel(r"$\|x^{(k)} - x^*\|$")
    if title:
        plt.title(title)
    plt.grid(True)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    # plt.show()

def Question1_solution():
    matrices = [Q_a, Q_b, Q_c, Q_d, Q_e]
    names = ['Q_a', 'Q_b', 'Q_c', 'Q_d', 'Q_e']
    

    b = np.array([1,1])
    x_initial = np.array([0,0])

    all_errors = []

    for i, Q in enumerate(matrices):
        print(f"\n{names[i]}")
        x, f_x, x_star, errors = exact_line_search(Q, b, x_initial, iter=10000)
        print("x:", x)
        print("f_x:", f_x)
        print("x_star_analytical:", x_star)

        all_errors.append(errors)

        # # To save the plots
        # savepath = f"{names[i]}_error_plot.png"
        # plot_errors(errors, title=f"Error plot for {names[i]}", savepath=savepath)

    # One figure 
    fig, axes = plt.subplots(1, 5, figsize=(20,4), sharey=True)
    for i, errors in enumerate(all_errors):
        iterations = range(len(errors))
        axes[i].plot(iterations, errors, marker="o")
        axes[i].set_title(names[i])
        axes[i].set_xlabel("Iter")
        if i == 0:
            axes[i].set_ylabel(r"$\|x^{(k)}-x^*\|$")
        axes[i].grid(True)

    plt.tight_layout()
    # plt.savefig("all_Q_error_plots.png", dpi=150, bbox_inches="tight")
    plt.show()

# Question1_solution()

# print(f"Condition numbers of Q_a {np.linalg.cond(Q_a)}")
# print(f"Condition numbers of Q_b {np.linalg.cond(Q_b)}")
# print(f"Condition numbers of Q_c {np.linalg.cond(Q_c)}")
# print(f"Condition numbers of Q_d {np.linalg.cond(Q_d)}")
# print(f"Condition numbers of Q_e {np.linalg.cond(Q_e)}")


# QUESTION 2

def armijo_condition(f_xk, grad_fxk, xk, alpha_0=1e-6, c=1e-2, iter=1000, beta=2):
    
    alpha = alpha_0
    last_alpha = alpha

    norm_grad_squared = (np.linalg.norm(grad_fxk))**2

    for i in range(iter):
        x_new = xk - alpha * grad_fxk

        f_x_new = oq2f(SR_No, x_new)

        # Increase alpha till Armijo condition is violated
        if f_x_new <= f_xk - c * alpha *  norm_grad_squared:
            last_alpha = alpha
            alpha *= beta
        else:
            return last_alpha
        
    return last_alpha

def armijo_goldstein_condition(f_xk, grad_fxk, xk, alpha_high=1, alpha_low=0, c=1e-4, iter=1000, beta=2):

    alpha = alpha_high
    norm_grad_squared = (np.linalg.norm(grad_fxk))**2

    for i in range(20):
        x_new = xk - alpha * grad_fxk

        f_x_new = oq2f(SR_No, x_new)

        if f_x_new <= f_xk - c * alpha * norm_grad_squared:
            alpha_high = alpha
            break
        alpha *= 2.0

    else:
        alpha_high = alpha

    # Bisection in [alpha_low, alpha_high]
    for i in range(iter):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        
        x_mid = xk - alpha_mid * grad_fxk
        f_x_mid = oq2f(SR_No, x_mid)

        lower = f_xk - (1 - c) * alpha_mid * norm_grad_squared
        upper = f_xk - c * alpha_mid * norm_grad_squared

        if lower <= f_x_mid <= upper:
            return alpha_mid
        elif f_x_mid < lower:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    return 0.5 * (alpha_low + alpha_high) 

# Helper for Wolfe_condition
def zoom_wolfe(f_xk, grad_fxk, xk, alpha_low, alpha_high, c1, c2, iter=20):
    
    norm_grad_squared = (np.linalg.norm(grad_fxk))**2

    for i in range(iter):
        alpha = 0.5 * (alpha_low + alpha_high)
        x_new = xk - alpha * grad_fxk
        f_x_new = oq2f(SR_No, x_new)
        f_x_alpha_low = oq2f(SR_No, xk - alpha_low * grad_fxk)

        # Armijo violation
        if (f_x_new > f_xk - c1 * alpha * norm_grad_squared) or (f_x_new >= f_x_alpha_low):
            alpha_high = alpha
        else:
            grad_f_x_new = oq2g(SR_No, x_new).reshape(-1) @ (-grad_fxk).reshape(-1)
            if abs(grad_f_x_new) <= c2 * norm_grad_squared:
                return alpha
            if grad_f_x_new * (alpha_high - alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha
    return alpha

def wolfe_condition(f_xk, grad_fxk, xk, alpha0=1.0, alpha_max=100, c1=1e-4, c2=0.8, iter=1000):
    
    norm_grad_squared = (np.linalg.norm(grad_fxk))**2

    alpha_prev = 0.0
    f_prev = f_xk

    alpha = alpha0
    for i in range(iter):
        x_new = xk - alpha * grad_fxk
        f_new = oq2f(SR_No, x_new)

        # Armijo violation 
        if (f_new > f_xk - c1 * alpha * norm_grad_squared) or (i > 0 and f_new >= f_prev):
            return zoom_wolfe(f_xk, grad_fxk, xk, alpha_prev, alpha, c1, c2)

        grad_f_x_new = oq2g(SR_No, x_new).reshape(-1) @ (-grad_fxk).reshape(-1)
        if abs(grad_f_x_new) <= c2 * norm_grad_squared:
            return alpha

        if grad_f_x_new >= 0:
            return zoom_wolfe(f_xk, grad_fxk, xk, alpha, alpha_prev, c1, c2)

        alpha_prev = alpha
        f_prev = f_new
        alpha = min(2*alpha, alpha_max) # Increasing alpha but keeping less than alpha_max

    return alpha

def backtracking(f_xk, grad_fxk, xk, alpha0=1.0, rho=0.5, c=1e-4, max_iter=100):

    alpha = alpha0
    norm_grad_squared = (np.linalg.norm(grad_fxk))**2

    for _ in range(max_iter):
        x_new = xk - alpha * grad_fxk
        f_x_new = oq2f(SR_No, x_new)

        # Checking Armijo sufficient decrease condition
        if f_x_new <= f_xk - c * alpha * norm_grad_squared:
            return alpha

        alpha *= rho

    return alpha

def gradient_descent(x0, iter=1000, tol=1e-10, method="armijo_condition"):

    x = np.array(x0).reshape(-1,1)
    # print(x.shape)
    # f_x_history = [] 
    alphas_history = []

    for i in range(iter):
        f_x = oq2f(SR_No, x)
        grad_fx = oq2g(SR_No, x)

        # f_x_history.append(f_x)

        if np.linalg.norm(grad_fx)<tol:
            print("ALGROITHM TERMINATED")
            break

        if method == "armijo_condition":
            alpha = armijo_condition(f_x, grad_fx, x)
            alphas_history.append(alpha)
            # print(alpha)
        elif method == "armijo_goldstein_condition":
            alpha = armijo_goldstein_condition(f_x, grad_fx, x)
            alphas_history.append(alpha)
            # print(alpha)
        elif method == "wolfe_condition":
            alpha = wolfe_condition(f_x, grad_fx, x)
            alphas_history.append(alpha)
            # print(alpha)
        elif method == "backtracking":
            alpha = backtracking(f_x, grad_fx, x)
            alphas_history.append(alpha)
            # print(alpha)
        else:
            print("Armijo Condition chosen by default")

        x = x - alpha * grad_fx
        # print(np.linalg.norm(grad_fx))
    
    f_x_star = oq2f(SR_No, x)
    grad_f_x_star = oq2g(SR_No, x)

    return x, f_x_star, grad_f_x_star, alphas_history


# Helper Global variables and functions for counting oracle calls
F_CALLS = 0
G_CALLS = 0
oq2f_original = None
oq2g_original = None

def reset_oracle_counts():
    global F_CALLS, G_CALLS
    F_CALLS = 0
    G_CALLS = 0

def enable_oracle_counting():

    global oq2f_original, oq2g_original, oq2f, oq2g

    if oq2f_original is None:
        oq2f_original = oq2f
    if oq2g_original is None:
        oq2g_original = oq2g

    reset_oracle_counts()

    def oq2f_wrapped(SR_No, x):
        global F_CALLS
        F_CALLS += 1
        return oq2f_original(SR_No, x)

    def oq2g_wrapped(SR_No, x):
        global G_CALLS
        G_CALLS += 1
        return oq2g_original(SR_No, x)

    # Modifying using the function so that it counts whenever they are called
    oq2f = oq2f_wrapped 
    oq2g = oq2g_wrapped

def Question2_solution():

    '''Descent Direction chosen -grad(f(x)) for all cases'''

    methods = [
    ("Armijo Condition", "armijo_condition"),
    ("Armijo-Goldstein Condition", "armijo_goldstein_condition"),
    ("Wolfe Condition", "wolfe_condition"),
    ("Backtracking", "backtracking")
    ]

    all_alphas = {}
    
    for method_name, method_key in methods:
        print(f"\n\n #### {method_name} \n\n")
        enable_oracle_counting()
        
        x, f_x_star, grad_f_x_star, alphas_history = gradient_descent(
            np.array([0,0,0,0,0]), 
            iter=1000, 
            method=method_key
        )
        
        print(f"x* : {x}")
        print(f"f(x*) : {f_x_star}")
        print(f"Gradient at x*: {grad_f_x_star}")
        print(f"Number of oracle 'oq2f' calls : {F_CALLS}")
        print(f"Number of oracle 'oq2g' calls : {G_CALLS}")
        
        all_alphas[method_name] = alphas_history
        
        # Plot for current method
        plt.figure()
        plt.plot(alphas_history, label='Step size (alpha)')
        plt.title(f'Step size history - {method_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Step size (alpha)')
        plt.legend()
        plt.grid(True)
        
        # For saving
        plt.savefig(f'{method_key}_alpha_history.png')
        plt.show()

    # Plot all together at last
    plt.figure(figsize=(10,6))
    for method_name, alphas in all_alphas.items():
        plt.plot(alphas, label=method_name)
    plt.title('Step size history comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Step size (alpha)')
    plt.legend()
    plt.grid(True)
    
    # For saving plot
    plt.savefig('all_methods_alpha_history_comparison.png')
    plt.show()

# Question2_solution()

# QUESTION 3

def armijo_condition_for_q3(f_xk, grad_fxk, xk, f, alpha_0=1e-6, c=1e-2, iter=1000, beta=2):
    
    alpha = alpha_0
    last_alpha = alpha

    norm_grad_squared = (np.linalg.norm(grad_fxk))**2

    for i in range(iter):
        x_new = xk - alpha * grad_fxk

        f_x_new = f(x_new)

        # Increase alpha till Armijo condition is violated
        if f_x_new <= f_xk - c * alpha *  norm_grad_squared:
            last_alpha = alpha
            alpha *= beta
        else:
            return last_alpha
        
    return last_alpha

# Objective Function (1/2) * ||Ax-b||^2
def oracle_for_objective_function(A, b):
    AT = A.T
    def f(x):
        r = A @ x - b
        return 0.5 * float(r.T @ r)
    def g(x):
        r = A @ x - b
        return AT @ r
    return f, g

def gradient_descent_for_q3(A, b, x0, tol=1e-10, iter=100):
    n = A.shape[1]
    x = np.zeros((n,1)) if x0 is None else np.array(x0, dtype=float).reshape(n,1) # Initial value all zeros if not provided

    f, g = oracle_for_objective_function(A, b)

    hist = []
    for i in range(iter):
        fx = f(x)
        gx = g(x)
        hist.append(fx)

        # print(fx)
        # print(gx)

        if np.linalg.norm(gx) <= tol:
            break
        alpha = armijo_condition_for_q3(fx, gx, x, f)
        # print(alpha)
        x = x - alpha * gx

    f_x_star = f(x)
    f_dash_x_star = g(x)
    return x, f_x_star, f_dash_x_star

def Question3_solution():
    A,b = oq3(24528)
    n = A.shape[1]

    x = np.zeros((n,1)).reshape(n,1)
    print(A.shape)
    print(b.shape)
    x, f_x_star, f_dash_x_star = gradient_descent_for_q3(A, b, np.zeros((n,1)), iter=400)

    print(f"Minima : {x}")
    print(f"Minimum Value : {f_x_star}")
    print(f"Gradient : {f_dash_x_star}")

    # Save minima x to CSV file (no header)
    np.savetxt('CMO2025A1_24528.csv', x, delimiter=',', fmt='%.8f')

# Question3_5
from time import perf_counter

def q3_5_comparison(min_power=1, max_power=16):
    np.random.seed(42)
    rows = []
    for s in range(min_power, max_power + 1):
        m = 2 ** s
        
        A = np.random.randn(m, m)

        b = np.random.randn(m, 1)

        # Using Inverse
        t0 = perf_counter()
        x_dir = np.linalg.solve(A, b)
        t1 = perf_counter()
        direct_solve_time = t1 - t0

        # Using optimization 
        t0 = perf_counter()
        x_opt, f_x_star, f_dash_x_star  = gradient_descent_for_q3(A, b, np.zeros((m,1)), iter=1000)
        t1 = perf_counter()
        optimization_solve_time = t1 - t0

        # residuals to compare
        delta_dir = np.linalg.norm(A @ x_dir - b) 
        delta_opt = np.linalg.norm(A @ x_opt - b) 

        rows.append((m, direct_solve_time, optimization_solve_time, delta_dir, delta_opt))
        print(f"m={m} | direct {direct_solve_time:0.8f}s | opt {optimization_solve_time:0.8f}s | residuals: direct {delta_dir:0.8f}, opt {delta_opt:0.8f}")


## Solutions, uncomment and run

# Question1_solution()
# Question2_solution()
# Question3_solution()
# q3_5_comparison(min_power=1, max_power=12)
