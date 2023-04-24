import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
%matplotlib notebook
# Define the dimensions of the rectangle
a, b = 1, 1

# Define the number of grid points
n, m = 51, 51

# Define the grid points
x = np.linspace(0, a, n)
y = np.linspace(0, b, m)

# Define the solution as given by the analytical expression
u = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        u[i,j] = np.sinh(np.pi*(1-y[j]))*np.sin(np.pi*x[i])*(1/np.sinh(np.pi))

# Plot the solution in 3D
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.contourf(X, Y, u, cmap='viridis')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()
