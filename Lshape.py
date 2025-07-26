import numpy as np
import matplotlib.pyplot as plt
import torch

# Define NLINEX loss function
def nlinex_loss(y_pred, y_true, k=2.5, c=0.5, alpha=1.0, beta=1.0):
    error = np.abs(y_true - y_pred)
    exp_term = alpha * (np.exp(k * error) - 1)
    quad_term = beta * error**2
    lin_term = c * error
    return exp_term + quad_term + lin_term

# Define baseline loss functions
def mse_loss(y_pred, y_true):
    return (y_true - y_pred)**2

def mae_loss(y_pred, y_true):
    return np.abs(y_true - y_pred)

def bce_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Avoid log(0)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Parameters for plotting
y_true = 0.25
y_pred = np.linspace(-1, 1, 1000)
nlinex_params = [
    {"k": 2.5, "c": 0.5, "label": "NLINEX (k=2.5, c=0.5)"},
    {"k": 2.0, "c": 0.4, "label": "NLINEX (k=2.0, c=0.4)"},
    {"k": 1.5, "c": 0.6, "label": "NLINEX (k=1.5, c=0.6)"}
]

# Compute losses
mse_values = [mse_loss(y, y_true) for y in y_pred]
mae_values = [mae_loss(y, y_true) for y in y_pred]
bce_values = [bce_loss(y, y_true) for y in y_pred]
nlinex_values = [
    [nlinex_loss(y, y_true, k=params["k"], c=params["c"]) for y in y_pred]
    for params in nlinex_params
]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(y_pred, mse_values, label="MSE", color="blue")
plt.plot(y_pred, mae_values, label="MAE", color="green")
plt.plot(y_pred, bce_values, label="BCE", color="red")
for i, values in enumerate(nlinex_values):
    plt.plot(y_pred, values, label=nlinex_params[i]["label"], linestyle="--")
plt.xlabel("Predicted Value")
plt.ylabel("Loss")
plt.title("Comparative Analysis of Loss Function Shapes (Ground Truth = 0.25)")
plt.legend()
plt.grid(True)
plt.savefig("figure1_loss_shapes.png", dpi=300)
plt.show()
