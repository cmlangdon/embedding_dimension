import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Data
# pseudo_response_tensor shape: (pseudo_trials, conditions, time, neurons)
pseudo_response_tensor = np.load('../ResponseData/CP_pseudo_responses.npy')
pseudo_response_tensor = pseudo_response_tensor[:, :, :, np.mean(pseudo_response_tensor, axis=(0, 1, 2)) < 15]
print(f"Loaded pseudo_response_tensor with shape: {pseudo_response_tensor.shape}")

# Reshape to (trials, time, neurons) where trials = pseudo_trials * conditions
# and keep track of the condition for each trial.
n_pseudo, n_conditions, n_time, n_neurons = pseudo_response_tensor.shape

# Flatten pseudo_trials and conditions into a single trials dimension
data_reshaped = pseudo_response_tensor.transpose(0, 1, 2, 3).reshape(n_pseudo * n_conditions, n_time, n_neurons)

# Trial conditions (second dimension of pseudo_response_tensor)
# For each trial in data_reshaped, we need its condition index
condition_indices = np.tile(np.arange(n_conditions), n_pseudo)

print(f"Data reshaped to: {data_reshaped.shape}")
print(f"Condition indices shape: {condition_indices.shape}")

# 2. Preprocessing
# Normalize data (Z-score per neuron across all trials and time)
data_mean = np.mean(data_reshaped, axis=(0, 1), keepdims=True)
data_std = np.std(data_reshaped, axis=(0, 1), keepdims=True) + 1e-8
data_norm = (data_reshaped - data_mean) / data_std

# Convert to torch tensor
y_train = torch.tensor(data_norm, dtype=torch.float32).to(device)

# 3. Create external inputs u
# Use the trial condition to create external inputs u.
# We'll map condition indices to a range, e.g., [-1, 1]
u_values = np.linspace(-1, 1, n_conditions)
u_train_np = u_values[condition_indices]  # shape (n_trials,)
u_train = torch.tensor(u_train_np, dtype=torch.float32).unsqueeze(1).to(device)  # shape (n_trials, 1)

print(f"External inputs u shape: {u_train.shape}")


# 4. Define RNN Model
class RNN(nn.Module):
    def __init__(self, n, input_dim, beta, bias, noise_std=0.0):
        super(RNN, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.beta = beta
        self.bias = bias
        self.noise_std = noise_std
        self.activation = nn.Softplus(beta=self.beta)

        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        # Initialize bias
        self.recurrent_layer.bias.data.normal_(mean=self.bias, std=0.01)

        self.input_layer = nn.Linear(self.input_dim, self.n, bias=False)
        # Use log_alpha for positivity and better scaling
        self.alpha = 0.1

    def forward(self, u, dt, steps):
        batch_size = u.shape[0]
        # Initial state: zero
        h0 = torch.zeros(batch_size, self.n).to(u.device)
        curr_state = h0
        all_states = [curr_state.unsqueeze(1)]

        input_contribution = self.input_layer(u)  # (batch, n)
        
        # Constrain alpha * dt to be between 0 and 1 for stability (Euler method)
        # Using a slightly smaller upper bound (0.9) for extra safety
        # alpha = 0.9 * torch.sigmoid(self.log_alpha) / dt

        for i in range(steps - 1):
            # Recurrent input
            rec_input = self.recurrent_layer(curr_state)  # (batch, n)

            # Total input to activation
            # dy/dt = -alpha * y + alpha * f(Wy + Bu + b)
            # To prevent Wy from exploding, we could also use an activation on it
            # but standard RNNs don't. Let's ensure curr_state doesn't explode.
            
            total_input = rec_input + input_contribution

            # Update state
            noise = 0
            if self.training and self.noise_std > 0:
                noise = torch.randn_like(curr_state) * self.noise_std

            # Euler integration: y(t+dt) = y(t) + dt * dy/dt
            # = y(t) + dt * (-alpha * y(t) + alpha * f(Wy(t) + Bu + b))
            # = (1 - alpha*dt) * y(t) + alpha * dt * f(total_input)
            
            next_state = (1 - self.alpha * dt) * curr_state + self.alpha * dt * self.activation(total_input) + noise
            
            # Robustness: clip state to prevent overflow
            curr_state = torch.clamp(next_state, min=-1e6, max=1e6)
            
            all_states.append(curr_state.unsqueeze(1))

        return torch.cat(all_states, dim=1)


# Model hyperparameters
n_hid = n_neurons
input_dim = 1
beta = 2
bias = 2
noise_std = 0.0
dt = 1.0
steps = n_time

model = RNN(n=n_hid, input_dim=input_dim, beta=beta, bias=bias, noise_std=noise_std).to(device)

# 5. Training Loop
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

epochs = 2000
batch_size = 32

print("Starting training...")
for epoch in range(epochs):
    # Simple batching
    permutation = torch.randperm(y_train.size(0))
    epoch_loss = 0

    for i in range(0, y_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_y = y_train[indices]
        batch_u = u_train[indices]

        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_u, dt, steps)

        # Z-score predictions
        pred_mean = predictions.mean(dim=(0, 1), keepdim=True)
        pred_std = predictions.std(dim=(0, 1), keepdim=True) + 1e-8
        
        # Guard against NaNs during normalization if predictions explode
        if torch.isnan(pred_std).any():
            print("NaN detected in pred_std, clipping predictions")
            predictions = torch.clamp(predictions, min=-1e6, max=1e6)
            pred_mean = predictions.mean(dim=(0, 1), keepdim=True)
            pred_std = predictions.std(dim=(0, 1), keepdim=True) + 1e-8

        predictions = (predictions - pred_mean) / pred_std

        # Loss
        loss = criterion(predictions, batch_y)
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / (y_train.size(0) / batch_size)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

torch.save(model, 'CP_net.pth')
