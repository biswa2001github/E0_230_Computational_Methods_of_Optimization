import os
import sys

sys.path.insert(0, os.path.abspath("oracle_2025A3"))
from oracle_2025A3 import f1  # type: ignore

# f1(24528)

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data_24528.csv')
beta_values = []

def Question_1():
    # df['x15_duplicate'] = df['x15']
    X = df.drop(columns=['y']).values
    y = df['y'].values

    m, n = len(df), len(df.columns) - 1

    print(m,n)

    def objective_function(X, y, beta, lambda_parameter):
        return (1/2) * cp.norm2(X @ beta - y)**2 + lambda_parameter * cp.norm1(beta)


    beta = cp.Variable(n)
    lambda_parameter = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(objective_function(X, y, beta, lambda_parameter)))

    lambda_values = [0.01, 0.1, 1]
    

    for i in lambda_values:
        lambda_parameter.value = i
        problem.solve()
        beta_values.append(beta.value)
        # print(i)

    print(beta_values[0])
    print(beta_values[1])
    print(beta_values[2])

    beta_values_array = np.array(beta_values)

    beta_df = pd.DataFrame(beta_values_array, columns=[f'beta_{i}' for i in range(beta_values_array.shape[1])])
    beta_df.insert(0, 'lambda', lambda_values)

    # beta_df.to_csv('lambda_beta_values_duplicate.csv', index=False)

    counts = np.sum(np.abs(beta_values_array) > 1e-6, axis=1)

    plt.figure()
    bars = plt.bar(range(len(lambda_values)), counts)

    plt.xticks(range(len(lambda_values)), lambda_values)
    plt.xlabel('λ')
    plt.ylabel('Number of non-zero coefficients')
    plt.title(r'Sparsity of $\beta$ vs $\lambda$ (Bar Chart)')

    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=9)


    # plt.savefig('sparsity_vs_lambda_duplicate.png', bbox_inches='tight', dpi=300)

    plt.show()

    # check 
    tol = 1e-4  # numerical tolerance

    for i, beta in enumerate(beta_values):
        lam = lambda_values[i]
        r = y - X @ beta

        print(f"\nChecking KKT condtions (Stationary conditions) for lambda = {lam}")

        for j in range(n):
            xj = X[:, j]
            lhs = xj.T @ r

            if abs(beta[j]) > 1e-6:
                rhs = lam * np.sign(beta[j])
                if abs(lhs - rhs) > tol:
                    print(f"KKT NOT satisfied for j={j} (non-zero beta). "
                        f"LHS={lhs:.6g}, RHS={rhs:.6g}, beta={beta[j]:.6g}")

            else:
                if abs(lhs) > lam + tol:
                    print(f"KKT NOT satisfied for j={j} (zero beta). "
                        f"LHS={lhs:.6g} should be in [-{lam/2.0}, {lam/2.0}]")

def Question_2():
    Question_1()
    X = df.drop(columns=['y']).values
    y = df['y'].values

    m, n = len(df), len(df.columns)

    # print(m,n)
    print("\n\n QUESTION 2 SOLUTION IS FROM HERE.\n\n")

    u = cp.Variable(m)
    lambda_values = [0.01, 0.1, 1]
  

    def dual_objective_function(u, y):
        return  - (1/2) * cp.norm2(u)**2 + y.T @ u
    
    u_values = []

    for i in lambda_values:
        lambda_parameter = i
        constraints = [cp.norm_inf(X.T @ u) <= lambda_parameter]
        problem = cp.Problem(cp.Maximize(dual_objective_function(u, y)), constraints)
        problem.solve()
        u_values.append(u.value)
    

    print(u_values[0])
    print(u_values[1])
    print(u_values[2])


    for idx, lam in enumerate(lambda_values):
        beta_star = beta_values[idx]
        u_star = u_values[idx]

        residual = y - X @ beta_star
        error_norm = np.linalg.norm(u_star - residual)

        print(f"Lambda = {lam}:  ||u* - (y - XB*)||_2 = {error_norm}")


## Please uncomment "Question_1()" or "Question_2()" to run
# Question_1()
# For executing Question2 I have including question_1 to run again so that the beta_values are fetched
# Question_2()


##### Following are the codes of Question 3
# For Question 3 
def PROJ_CIRCLE(y, center=np.array([0,0]), radius=5):
    y_center = y - center
    norm_y_center = np.linalg.norm(y_center)
    if norm_y_center <= radius:
        y_proj = y
    else:
        y_proj = center + (radius / norm_y_center) * y_center
    return y_proj

def PROJ_BOX(y, low=np.array([-3,0]), high=np.array([3,4])):
    if y[0] < low[0]:
        x1 = low[0]
    elif y[0] > high[0]:
        x1 = high[0]
    else:
        x1 = y[0]

    if y[1] < low[1]:
        x2 = low[1]
    elif y[1] > high[1]:
        x2 = high[1]
    else:
        x2 = y[1]

    y_proj = np.array([x1, x2])

    return y_proj

# print(PROJ_BOX(np.array([7,2])))

def plot_projections():
    samples = np.array([[7,2],[2,2],[-6,-3],[0,6],[4,-1]])

    plt.figure()
    center = np.array([0.,0.])
    r = 5
    circle = plt.Circle(center, r, color='lightblue', alpha=0.3)
    ax = plt.gca()
    ax.add_patch(circle)
    for y in samples:
        proj = PROJ_CIRCLE(y)
        plt.plot([y[0], proj[0]], [y[1], proj[1]], 'r--')
        ax.plot(*y, 'ro')
        ax.plot(*proj, 'go')

    plt.title("Projection onto Circle")

    plt.xlim(-8,8)
    plt.ylim(-8,8)

    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig("projection_circlee.png", dpi=300)

    plt.figure()
    low = np.array([-3,0])
    high = np.array([3,4])
    rect = plt.Rectangle(low, high[0]-low[0], high[1]-low[1], color='lightgreen', alpha=0.3)
    ax = plt.gca()
    ax.add_patch(rect)
    for y in samples:
        proj = PROJ_BOX(y)
        plt.plot([y[0], proj[0]], [y[1], proj[1]], 'r--')
        ax.plot(*y, 'ro')
        ax.plot(*proj, 'go')
    plt.title("Projection onto Rectangle")
    plt.xlim(-8,8)
    plt.ylim(-4,8)
    plt.gca().set_aspect('equal')
    plt.grid(True)

    plt.savefig("projection_box.png", dpi=300)

    plt.show()


# plot_projections()

def Questionn_3_1():
    plot_projections()

# Questionn_3_1()

def C_A(x):
    '''This is projection of any point on set A which is circle of unit radius in this case'''
    norm = np.linalg.norm(x)

    if norm <= 1:
        return x
    else:
        return x / norm

def C_B(x):
    y = x.copy()
    if y[0]<3:
        y[0] = 3
    return y

def SEPERATE_HYPERPLANE(C_A, C_B):
    max_iter = 100
    delta = 1e-6

    x = np.zeros(2, dtype=float)

    for i in range(max_iter):
        y = C_A(x)

        x_dash = C_B(y)

        if np.linalg.norm(x_dash - x) < delta:
            break

        x = x_dash
    
    a_closest = y
    b_closest = x_dash

    n = b_closest - a_closest

    c = n @ ((a_closest + b_closest)/2)

    return n, c, a_closest, b_closest

def Question_3_2():
    n, c, a_closest, b_closest = SEPERATE_HYPERPLANE(C_A, C_B)

    print(n, c, a_closest, b_closest)

# Question_3_2()

def CHECK_FARKAS():
    A = np.array([[1,1],
                  [-1, 0],
                  [0, -1]])

    b = np.array([-1, 0, 0])

    m,n = A.shape
    
    x = cp.Variable(n)
    # Giving a constant number like 100 in this case
    problem = cp.Problem(cp.Minimize(100), [A @ x <= b])
    problem.solve()

    feasibility = problem.status
    # print(feasibility)
    if feasibility == "optimal":
        return True, None, problem.status

    # Farkascertifcate
    y = cp.Variable(m)
    # cp.sum(y)==1 fornormalizing
    problem_certificate = cp.Problem(cp.Minimize(b.T @ y), [A.T @ y == 0, y >= 0, cp.sum(y)==1])
    problem_certificate.solve()

    y_cert = np.array(y.value).reshape(-1)

    diagnostic_info = {
        "feasibility_status": problem.status,
        "certificate_status": problem_certificate.status,
        "cert_obj": float(problem_certificate.value)
    }

    return False, y_cert, diagnostic_info

def Question_3_3():
    feasible, y_cert, diagnostic_info = CHECK_FARKAS()
    print(feasible, y_cert, diagnostic_info)

# Question_3_3()