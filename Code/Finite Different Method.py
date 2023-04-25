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
u_fdm = np.zeros((n, m))
u_fdm[0, :] = 0
u_fdm[-1, :] = 0
u_fdm[:, 0] =  np.sin(np.pi*x)
u_fdm[:, -1] = 0               

# Define the tolerance and maximum number of iterations
tol = 1e-4
maxiter = 10000

# Define the update function
def update(u):
    unew = np.copy(u_fdm)
    for i in range(1, n-1):
        for j in range(1, m-1):
            unew[i, j] = 0.25*(u_fdm[i+1, j] + u_fdm[i-1, j] + u_fdm[i, j+1] + u_fdm[i, j-1])
    return unew

# Solve the Laplace equation using the Finite Difference Method
for k in range(maxiter):
    unew = update(u_fdm)
    if np.linalg.norm(unew - u_fdm) < tol:
        u = unew
        break
    u_fdm = unew

# Plot the solution in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, u_fdm, cmap='viridis')
ax.plot_surface(X, Y, u_fdm, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.title('Solution to Laplace Equation')
plt.show()

print(np.linalg.norm(u- u_fdm))
