import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook

# Define the dimensions of the rectangle
a, b = 1, 1

# Define the number of grid points
n, m = 51, 51


# Define the grid points
x = np.linspace(0, a, n)
y = np.linspace(0, b, m)

# Define the initial solution with Dirichlet boundary conditions
u = np.zeros((n, m))
u[0, :] = 0  
u[-1, :] = 0
u[:, 0] = np.sin(np.pi*x)
u[:, -1] = 0               

# Define the tolerance and maximum number of iterations
tol = 1e-4
maxiter = 10000

# Define the update function
def update(u):
    unew = np.copy(u)
    for i in range(1, n-1):
        for j in range(1, m-1):
            unew[i, j] = 0.25*(u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
    return unew

# Solve the Laplace equation using the Finite Difference Method
for k in range(maxiter):
    unew = update(u)
    if np.linalg.norm(unew - u) < tol:
        u = unew
        break
    u = unew

# Plot the solution in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()

plt.contourf(X, Y, u.T, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution to Laplace Equation')
plt.show()
