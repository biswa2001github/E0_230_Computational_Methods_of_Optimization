import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("oracle_2025A0"))
from oracle_2025A0 import oracle  # type: ignore

def gradient_descent(x0, SR_No, lr=0.001, steps=100, tol=1e-8):
    '''

    Arguments
    - x0 : Starting initial value
    - SR_No : SR Number
    - lr : Learning rate for the gradient descent algorithm (default is 0.001)
    - steps : Maximum number of iterations to perform
    - tol : Tolerance for the first derivative in this case (default is 1e-8) 
        e.g if the (absolute value of first derivative) <= tol then we stop the algorithm before the max steps 
        completes as we assume it converges as the first derivative value is very close to zero

    Returns
    - x : Minima calculated the the algorithm
    - f_x_star : Minimum value
    - f_dash_x_star : First derivate value at the minima
    - f_x_hist : The f_x values at each stage for plotting or visualization purpose
    '''

    x = x0
    f_x_history = [] 

    for i in range(steps):
        f_x, f_dash_x = oracle(SR_No, x)
        f_x_history.append(f_x)

        if abs(f_dash_x)<tol:
            break

        x = x - lr * f_dash_x
    
    f_x_star, f_dash_x_star = oracle(SR_No, x)

    return x, f_x_star, f_dash_x_star, f_x_history

def gradient_descent_with_momentum(x0, SR_No, lr=0.001, steps=100, momentum=0.9, tol=1e-6):
    '''

    Arguments
    - x0 : Starting initial value
    - SR_No : SR Number
    - lr : Learning rate for the gradient descent algorithm (default is 0.001)
    - steps : Maximum number of iterations to perform
    - momentum : Momentum parameter ranges from 0 to 1 (default is set to 0.9)
    - tol : Tolerance for the first derivative in this case (default is 1e-8) 
        e.g if the (absolute value of first derivative)<=tol then we stop the algorithm before the max steps 
        completes as we assume it converges as the first derivative value is very close to zero

    Returns
    - x : Minima calculated the the algorithm
    - f_x_star : Minimum value
    - f_dash_x_star : First derivate value at the minima
    - f_x_history : f_x values at each stage for plotting purpose
    '''

    x = x0
    v = 0 # This is the velocity value set to zero 
    f_x_history = []

    for i in range(steps):
        f_x, f_dash_x = oracle(SR_No, x)
        f_x_history.append(f_x)

        if abs(f_dash_x)<tol:
            break
        
        v = momentum * v + lr * f_dash_x
        x = x - v
    
    f_x_star, f_dash_x_star = oracle(SR_No, x)

    return x, f_x_star, f_dash_x_star, f_x_history

def compare_plots():
    # Finding minima using Gradient Descent
    x_star, f_x_star, f_dash_x_star, f_x_history_gd = gradient_descent(x0=1, SR_No=24528, lr=0.01, steps=200)

    # Finding minima using Gradient Descent with momentum
    x_star, f_x_star, f_dash_x_star, f_x_history_gdm = gradient_descent_with_momentum(x0=1, SR_No=24528, lr=0.01, steps=200, momentum=0.9)

    # Plots for comparing gradient descent and gradient descent with momentum
    plt.figure(figsize=(8, 5))
    plt.plot(f_x_history_gd, label='Gradient Descent')
    plt.plot(f_x_history_gdm, label='Gradient Descent with Momentum')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title('Values of f(x) with number of iterations')
    plt.grid()
    plt.show()

# Finding minima using Gradient Descent with momentum
x_star, f_x_star, f_dash_x_star, f_x_history_gdm = gradient_descent_with_momentum(x0=1, SR_No=24528, lr=0.01, steps=1000, momentum=0.9)
print(f"\nUSING GRADIENT DESCENT WITH MOMENTUM:\nMinima(x*) = {x_star}\nMinimum_value(f(x*)) = {f_x_star}\nFirst Derivative at (x=x*) = {f_dash_x_star}\n")

# Function to view the comparison on convergence
# compare_plots() # Uncomment for viewing plots