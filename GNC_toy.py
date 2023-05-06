import numpy as np
import matplotlib.pyplot as plt

# Define the objective function
def objective(x):
    return (x ** 4) - (4 * x ** 2) + (x ** 2) / 2

# Define the gradient of the objective function
def gradient(x):
    return 4 * (x ** 3) - 8 * x + x

# Define the Hessian matrix of the objective function
def hessian(x):
    return 12 * (x ** 2) - 8

# Perform gradient descent optimization
def gradient_descent(learning_rate, num_iterations):
    x = np.random.uniform(-2, 2)  # Initialize x randomly
    trajectory = [x]

    for _ in range(num_iterations):
        grad = gradient(x)
        x -= learning_rate * grad
        trajectory.append(x)

    return x, trajectory

# Perform Newton's method optimization
def newtons_method(num_iterations):
    x = np.random.uniform(-2, 2)  # Initialize x randomly
    trajectory = [x]

    for _ in range(num_iterations):
        grad = gradient(x)
        hess = hessian(x)
        x -= grad / hess
        trajectory.append(x)

    return x, trajectory

# Main code
learning_rate = 0.1
num_iterations = 50

# Run gradient descent optimization
gd_solution, gd_trajectory = gradient_descent(learning_rate, num_iterations)

# Run Newton's method optimization
newton_solution, newton_trajectory = newtons_method(num_iterations)

# Plot the objective function and the optimization trajectories
x_vals = np.linspace(-2, 2, 100)
y_vals = objective(x_vals)

plt.plot(x_vals, y_vals, label='Objective Function')
plt.plot(gd_trajectory, [objective(x) for x in gd_trajectory], '-o', label='Gradient Descent')
plt.plot(newton_trajectory, [objective(x) for x in newton_trajectory], '-o', label="Newton's Method")
plt.xlabel('x')
plt.ylabel('Objective Value')
plt.legend()
plt.title('Graduated Non-Convexity')
plt.show()