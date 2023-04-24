import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook
# Define the dimensions of the rectangle
a, b = 1, 1

# Define the number of grid points
n, m = 51, 51

# Define the step sizes
dx, dy = a/(n-1), b/(m-1)

# Define the grid points
x = np.linspace(0, a, n)
y = np.linspace(0, b, m)

# Define the input and output data for the neural network
x_data = np.vstack((np.tile(x, m), np.repeat(y, n)))
y_data = u.reshape(-1, 1)

# Define the neural network architecture
input_layer = tf.keras.layers.Input(shape=(2,))
hidden_layer_1 = tf.keras.layers.Dense(32, activation='relu')(input_layer)
hidden_layer_2 = tf.keras.layers.Dense(32, activation='relu')(hidden_layer_1)
output_layer = tf.keras.layers.Dense(1, activation=None)(hidden_layer_2)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Define the loss function and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Train the neural network
for i in range(2000):
    with tf.GradientTape() as tape:
        y_pred = model(x_data.T)
        loss = loss_fn(y_data, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.numpy():.4f}")

# Evaluate the neural network
u_pred = model(x_data.T).numpy().reshape(n, m)

# Plot the solution
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.contourf(X, Y, u_pred, cmap='viridis')
ax.plot_surface(X, Y, u_pred, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()
print(np.linalg.norm(u- u_pred))
