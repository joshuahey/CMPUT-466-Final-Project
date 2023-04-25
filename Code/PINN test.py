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

# Split the data into training, validation, and testing sets
n_train = int(0.70 * n * m)
n_val = int(0.15 * n * m)
n_test = n * m - n_train - n_val
idx = np.random.permutation(n * m)
idx_train = idx[:n_train]
idx_val = idx[n_train:n_train+n_val]
idx_test = idx[n_train+n_val:]
x_train, y_train = x_data[:, idx_train], y_data[idx_train]
x_val, y_val = x_data[:, idx_val], y_data[idx_val]
x_test, y_test = x_data[:, idx_test], y_data[idx_test]

# Define the neural network architecture
input_layer = tf.keras.layers.Input(shape=(2,))
hidden_layer_1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)
hidden_layer_2 = tf.keras.layers.Dense(64, activation='relu')(hidden_layer_1)
output_layer = tf.keras.layers.Dense(1, activation=None)(hidden_layer_2)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

def residual(u, x, y, dx, dy):
    u_xx = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
    u_yy = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    r = u_xx + u_yy

    return r.reshape(-1, 1)

def residual_loss(u, x, y, dx, dy):
    r = residual(u, x, y, dx, dy)
    boundary_loss = np.mean((u[:, 0] - np.sin(np.pi*x))**2)  # Add boundary loss
    return np.mean(r**2) + boundary_loss



# Train the neural network
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(x_data.T)
        loss = loss_fn(y_data, y_pred) + residual_loss(u, x, y, dx, dy)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss.numpy():.4f}")

# Evaluate the neural network on the training set
u_train = model(x_train.T).numpy().reshape(-1, 1)
train_loss = loss_fn(y_train, u_train)
print(f"Training Loss: {train_loss.numpy():.4f}")


# Evaluate the neural network on the validation set
u_val = model(x_val.T).numpy().reshape(-1, 1)
val_loss = loss_fn(y_val, u_val)
print(f"Validation Loss: {val_loss.numpy():.4f}")

# Evaluate the neural network on the testing set
u_test = model(x_test.T).numpy().reshape(-1, 1)
test_loss = loss_fn(y_test, u_test)
print(f"Testing Loss: {test_loss.numpy():.4f}")

X_test, Y_test = np.meshgrid(x_test, y_test)

# Create a meshgrid for the entire domain
X, Y = np.meshgrid(x, y)

# Create a meshgrid for the predicted u values on the testing set
U_test = np.zeros((n, m))
U_test[idx_test // m, idx_test % m] = u_test.reshape(-1)
# Plot the entire domain with the predicted u values on the testing set
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U_test, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('Entire Domain with Predicted u Values on Test Set')
plt.show()
