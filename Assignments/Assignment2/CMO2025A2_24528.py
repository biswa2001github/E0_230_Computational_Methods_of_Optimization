import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("oracle_CMO2025A2_py310"))
from oracle_CMO2025A2_py310.oracle_final_CMOA2 import f2, f5

'''
Please comment out the functions termed as e.g "Question_1_1_c()" to run the solution for respective questions
'''

#### QUESTION 1

Q, b = f2(24528, True)

def CD_SOLVE(A, b, x0=np.zeros(np.shape(b)), maxiter=1000):
    x = x0
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Lists
    alphas = []
    numerators = []
    lambdas = []

    # print(eigenvalues)
    
    for i in range(maxiter):
        g = A @ x - b
        print(np.linalg.norm(g))
        u = eigenvectors[:, i]

        gTu = - g.T @ u

        alpha = gTu / (u.T @ A @ u)

        alphas.append(float(alpha))
        numerators.append(float(gTu))
        lambdas.append(float(eigenvalues[i]))

        # print(f"Alpha: {alpha}, -grad f(x)^T u: {gTu}, lambda: {eigenvalues[i]}")

        x = x + alpha * u

    return x, alphas, numerators, lambdas

def CG_SOLVE(A, b, tol=1e-6, maxiter=10000, log_directions=False, use_relative_tol=False):
    # Initialization
    x = np.zeros(np.shape(b))
    g = A @ x - b
    u = - g

    r0 = -g

    residual_list = []
    directions = []
    residuals = [np.linalg.norm(r0)]
    
    if log_directions:
        residual_list.append(b - A @ x)
        directions.append(u)

    for i in range(maxiter):
        # print(f"\nITERATION : {i+1}")
        gTu = - g.T @ u
        # print(f"-grad f(x)^T u : {gTu}")

        alpha = gTu/(u.T @ A @ u)
        # print(f"alpha : {alpha}")

        x = x + alpha * u
        # print(f"x{i+1} : {x}")

        g = g + alpha * (A @ u)
        B = (g.T @ A @ u)/(u.T @ A @ u)
        u = -g + B * u

        r = b - A @ x
        r_norm = np.linalg.norm(r)
        residuals.append(float(r_norm))

        if use_relative_tol:
            if (r_norm / np.linalg.norm(r0)) < tol:
                break
        else:
            if r_norm < tol:
                break

        if log_directions:
            residual_list.append(r)
            directions.append(u)

    if log_directions:
        return x, i+1, residuals, residual_list, directions
    else:
        return x, i+1, residuals
 
def Question_1_1c():
    x, alphas, numerators, lambdas = CD_SOLVE(A=Q, b=b, x0=np.zeros(np.shape(b)), maxiter=7)

    # print(x)
    # print(alphas)
    # print(numerators)
    # print(lambdas)

    for k in range(len(alphas)):
        print(f"({alphas[k]}, {numerators[k]}, {lambdas[k]})")

def Question_1_2():
    x, iters, residuals, r_list, p_list = CG_SOLVE(A=Q, b=b, log_directions=True)

    print(x)
    print("\n")
    print(f"Number of iterations: {iters}")
    # print(len(residuals))
    # print(len(r_list))
    print(f"Number of m directions computed: {len(p_list)}")

    # print(r_list[0])
    # Convert lists of vectors to arrays
    plist = np.array(p_list)  # shape: (num_iters, n)
    np.savetxt(f"plist_24528.txt", plist, fmt="%.10e")
    
# Helper function for GS_ORTHOGONALISE
def summation(P_k, Q, k, D):
    summation_value = np.zeros(np.shape(D[0]))
    
    for i in range(k+1):
        summation_value += ((P_k.T @ Q @ D[i]) / (D[i].T @ Q @ D[i])) * D[i]
    return summation_value

def GS_ORTHOGONALISE(P, A):

    D = [P[0]]
    for i in range(len(P)-1):
        D_new = P[i+1] - summation(P[i+1],Q,k=i, D=D)
        D.append(D_new)
    return D

def normalize_dk(D, A):
    D_tilde = []
    for d in D:
        norm_A = np.sqrt(float(d.T @ A @ d))
        D_tilde.append(d / norm_A)
    return D_tilde

def form_matrix_M(D, A):
    n = len(D)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            entry_ij = float(D[i].T @ A @ D[j])
            M[i, j] = entry_ij
    return M

def Question_1_1_3():
    _, _ ,_ ,_ ,p_list = CG_SOLVE(A=Q, b=b, log_directions=True)

    D = GS_ORTHOGONALISE(P = p_list, A = Q)

    D_array = np.array(D)
    filename = f"dlist_24528.txt"
    np.savetxt(filename, D_array, fmt="%.10e")

    # Normalize D and make M
    D_tilde = normalize_dk(D, Q)
    M = form_matrix_M(D_tilde, Q)

    # Using this for properly putting this matrix in report and also appplying a tolerance for printing in consise manner
    def matrix_to_latex(M, tol=1e-16, precision=2):
        M = np.where(np.abs(M) < tol, 0.0, M)

        def format(v):
            if v == 0:
                return "0"
            else:
                return f"{v:.{precision}e}"

        rows = [" & ".join(format(v) for v in row) + r" \\" for row in M]
        return "\\begin{bmatrix}\n" + "\n".join(rows) + "\n\\end{bmatrix}"


    # Example usage:
    print(matrix_to_latex(M, tol=1e-16))

def compare_Q_Gram_Schmidt_with_CG():
    Q, b = f2(24528, True)
    _, _ ,_ ,_ ,p_list = CG_SOLVE(A=Q, b=b, log_directions=True)
    P = p_list

    D = GS_ORTHOGONALISE(P = p_list, A = Q)
    
    n = len(D)
    cos_theta = []

    for i in range(n):
        val = (P[i].T @ Q @ D[i]) / (np.sqrt(P[i].T @ Q @ P[i]) * np.sqrt(D[i].T @ Q @ D[i]))
        cos_theta.append(float(val))

    return cos_theta

def Question_1_1_4():
    cos_thetas = compare_Q_Gram_Schmidt_with_CG()

    print(cos_thetas)

# Question_1_1c()
# Question_1_2()
# Question_1_1_3()
# Question_1_1_4()


#### QUESTION 2


A, b = f5(24528)

def plot_residual_norm_vs_iterations(residuals, savepath):

    plt.figure(figsize=(6,4))
    plt.plot(range(len(residuals)), residuals, marker='o')
    # plt.yscale("log")   # for checking with log scale if needed
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Conjugate Gradient Residual Norm vs Iterations")
    plt.grid(True)
    plt.savefig(savepath)
    plt.show()

def Question_2_1():
    x, iters, residuals = CG_SOLVE(A, b, log_directions=False, use_relative_tol=True)
    print(f"Number of iterations: {iters}")
    plot_residual_norm_vs_iterations(residuals, savepath='CG_SOLVE_PLOT.png')


def Jacobi_preconditioner(A, n):
    diagonal = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        Ae = A @ e  
        diagonal[i] = Ae[i]
    return diagonal

def CG_SOLVE_FAST(A, b, tol=1e-6, maxiter=10, log_directions=False):
    
    n = len(b)
    x = np.zeros(np.shape(b))

    diagonal_A = Jacobi_preconditioner(A, n)
    M_inverse = 1.0 / diagonal_A

    g = A @ x - b
    y = M_inverse * g
    u = -y

    residuals = [np.linalg.norm(g)]
    residual_list = []
    directions = []

    r0_norm = np.linalg.norm(b - A @ x)

    if log_directions:
        residual_list.append(r0_norm)
        directions.append(u)

    for k in range(maxiter):
        gTy = g.T @ y
        alpha = gTy / (u.T @ A @ u)

        x = x + alpha * u
        g = g + alpha * (A @ u)

        r = b - A @ x

        r_norm = np.linalg.norm(r)
        residuals.append(float(r_norm))

        if (r_norm / r0_norm) < tol:
            break

        y_new = M_inverse * g

        beta = (g.T @ y_new) / gTy

        u = -y_new + beta * u

        y = y_new

        if log_directions:
            residual_list.append(r)
            directions.append(u)

    if log_directions:
        return x, k, residuals, residual_list, directions
    else:
        return x, k, residuals

def Question_2_2():
    x, iters, residuals = CG_SOLVE_FAST(A, b, log_directions=False)
    x1, _, _ = CG_SOLVE(A, b, log_directions=False)
    # print(np.linalg.norm(b-A@x))
    plot_residual_norm_vs_iterations(residuals, savepath='CF_SOLVE_FAST.png')

Question_2_1()
# Question_2_2()






#### QUESTION 3

def NEWTON_SOLVE(f_grad, f_hess, x0, tol=1e-8, maxiter=100):
    x = x0
    g = f_grad(x)
    h = f_hess(x) 

    trajectory = [x]

    for i in range(maxiter):
        if (np.linalg.norm(g) < tol):
            # print(np.linalg.norm(g))
            break

        # print(np.linalg.norm(g))
        p = np.linalg.solve(h,-g)

        x = x + p
        trajectory.append(x)

        g = f_grad(x)
        h = f_hess(x)

    return x, i, trajectory

def f_grad(x):
    x1, x2 = x

    df_dx1 = 400 * (x1**3) - 400 * x1 * x2 + 2 * x1 - 2
    df_dx2 = 200 * x2 - 200 * (x1**2)

    return np.array([df_dx1, df_dx2])

def f_hess(x):
    x1, x2 = x

    d2f_dx1_2 = 1200 * (x1**2) - 400 * x2 + 2
    d2f_dx1_dx2 = -400 * x1
    d2f_dx2_2 = 200

    return np.array([[d2f_dx1_2, d2f_dx1_dx2],
                     [d2f_dx1_dx2, d2f_dx2_2]])

def f(x):
    x1, x2 = x
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def analysis():
    starting_points = [
        np.array([2, 2]),
        np.array([5, 5]),
        np.array([-10, -4]),
        np.array([50, 60])
    ]

    X = np.linspace(-10, 10, 400)
    Y = np.linspace(-10, 10, 400)
    XX, YY = np.meshgrid(X, Y)
    Z = 100 * (YY - XX**2)**2 + (1 - XX)**2

    x_star = np.array([1, 1])
    error_curves = {}

    for x0 in starting_points:
        _, iter, traj = NEWTON_SOLVE(f_grad, f_hess, x0)
        print(f"Iterations for {x0} : {iter}")
        traj = np.array(traj)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.contour(XX, YY, np.log10(Z + 1), levels=30)
        ax.plot(traj[:, 0], traj[:, 1], 'r-o', markersize=4, label='Newton iterates')
        ax.plot(1, 1, 'b*', markersize=12, label='Min (1,1)')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_title(f"Newton Method Trajectory (Start: {x0.tolist()})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend()
        ax.grid(True, ls="--", lw=1)
        fig.tight_layout()

        filename = f"newton_trajectory_start_{x0[0]}_{x0[1]}.png"
        fig.savefig(filename, dpi=300)
        plt.close(fig)

        print(f"Saved: {filename}")

        errors = [np.linalg.norm(x - x_star) for x in traj]
        error_curves[tuple(x0)] = errors
    
    plt.figure(figsize=(8, 6))
    for start, errors in error_curves.items():
        plt.plot(errors, marker='o', label=f"Start {(float(start[0]), float(start[1]))}")

    # plt.yscale('log') # for debugging used this
    plt.xlabel("Iteration number k")
    plt.ylabel(r"$\|x_k - x^*\|_2$")
    plt.title("Error vs Iteration for Newton's Method")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("newton_error_vs_iteration.png", dpi=300)
    plt.close()
    print("Saved: newton_error_vs_iteration.png")

def Question_3():
    analysis() 

# Question_3()



#### MISCELLANEOUS : if required comment out specific parts to run

## Newtons method for very far point
# x, iter, traj = NEWTON_SOLVE(f_grad, f_hess, x0=np.array([1e10,1e20]))


## For checking A is diagonal or not
# A, b = f5(24528)

# from scipy.sparse.linalg import LinearOperator

# def is_near_diagonal(A: LinearOperator, tol=1e-6):
#     n = A.shape[1]

#     print(n)
#     off_diag_ratio = []
#     for i in range(n):
#         e = np.zeros(n)
#         e[i] = 1.0
#         col = A @ e

#         diag_val = col[i]

#         off_diag_norm = np.linalg.norm(np.delete(col, i))
#         diag_norm = abs(diag_val)

#         ratio = off_diag_norm / (diag_norm + 1e-15)
#         off_diag_ratio.append(ratio)

#     avg_ratio = np.mean(off_diag_ratio)
#     return avg_ratio < tol, avg_ratio

# print(is_near_diagonal(A))