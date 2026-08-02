import pandas as pd
import scipy.stats
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
import torch
import os
import sys
import umap

# Add project root to path
sys.path.append(os.path.abspath('../../'))
import torch.optim as optim
from bwm_functions import *

import torch
import torch.nn as nn
import torch.optim as optim
plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .5


dt = 1
steps = 100

top_k = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Define RNN class
class RNN(torch.nn.Module):
    def __init__(self, n, input_dim, beta, bias):
        super(RNN, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.beta = beta
        self.bias = bias
        self.activation = torch.nn.Softplus(beta=self.beta)
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        self.recurrent_layer.bias.data.normal_(mean=self.bias, std=0)
        self.recurrent_layer.bias.requires_grad = True
        self.input_layer = nn.Linear(self.input_dim, self.n, bias=False)
        self.alpha = nn.Parameter(.1*torch.ones(n))
    def forward(self, u, dt, steps):
        batch_size = u.shape[0]
        states = torch.zeros(batch_size, 1, self.n, device=device)
        for i in range(steps - 1):
            state_new = (1 - self.alpha * dt) * states[:, i, :] + self.alpha *dt * (
                self.activation(self.recurrent_layer(states[:, i, :]) + self.input_layer(u)))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states


# Load trained RNN
u = torch.FloatTensor(np.linspace(-1, 1, 2)).to(device)
net = torch.load('CP_net.pth', weights_only=False, map_location=device).to(device)


# Fix and parameterize DensMAP projection
y = net(u=u.unsqueeze(1), dt=dt, steps=100).detach()
z = (y - torch.mean(y, dim=(0, 1))) / torch.std(y, dim=(0, 1))
x = z.reshape(-1, z.shape[2]).t()
# umap doesn't support cuda directly in the standard library, so we keep it on cpu then move result to device
y = torch.FloatTensor(umap.UMAP(densmap=True, dens_lambda=2, n_neighbors=500).fit_transform(x.cpu())).to(device)

class TensorTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(TensorTransformer, self).__init__()
        # A simple 3-layer architecture with dropout for robustness
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),    # Input layer to hidden layer 1
            nn.ReLU(),                   # Non-linear activation
            nn.Dropout(dropout_rate),    # Dropout for regularization
            nn.Linear(64, 32),           # Hidden layer 1 to hidden layer 2
            nn.ReLU(),                   # Non-linear activation
            nn.Dropout(dropout_rate),    # Dropout for regularization
            nn.Linear(32, output_dim)    # Hidden layer 2 to output layer
        )

    def forward(self, x, noise_std=0.01):
        # Add Gaussian noise to input during training for better generalization
        if self.training and noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        return self.network(x)


# Parameterize densmap projection
epochs = 10000
learning_rate = 0.001
densMAPprojector = TensorTransformer(input_dim=x.shape[1], output_dim=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(densMAPprojector.parameters(), lr=learning_rate)

# Put model in training mode
densMAPprojector.train()

for epoch in range(epochs):
    # 1. Forward pass: compute predicted y by passing x to the model
    predictions = densMAPprojector(x)

    # 2. Compute loss
    loss = criterion(predictions, y)

    # 3. Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Gradient descent step

    # Print progress every 20 epochs
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


def efficient_variance_explained_svd_vals(responses,top_k):
    x_centered = responses - responses.mean(dim=(0),keepdim=True)
    S = torch.linalg.svdvals(x_centered)
    eigenvalues = S**2
    actual_k = min(top_k,len(eigenvalues))
    return eigenvalues[:actual_k]/eigenvalues.sum()



# Allocate lists for projections and speed
projections = []
speeds = []
initial_projection = None
# Contract RNN
net_contract = torch.load('CP_net.pth', weights_only=False, map_location=device).to(device)
u = torch.FloatTensor(np.linspace(-1, 1, 2)).to(device)
optimizer = optim.Adam(net_contract.parameters(), lr=.0001)
criterion = nn.MSELoss()

#curvature_target = fsecond(torch.linspace(0,100,100),shift= 20, scale = 2,height = 3)
for epoch in range(10000):
    optimizer.zero_grad()
    x = net_contract(u.unsqueeze(1),dt,steps)


    # Activity penalty
    mean_activities = x.mean(dim=(0, 1))
    activity_penalty = mean_activities.std()

    # Variance penalty
    fve = efficient_variance_explained_svd_vals(x.reshape(-1, x.shape[2]).t(),top_k = top_k)
   # variance_penalty = torch.mean((fve-target_fve)**2)
    variance_penalty = torch.mean(fve[3:])

    # Speed penalty
    z = (x-torch.mean(x,dim=(0,1))) / torch.std(x,dim=(0,1))
    speed = torch.mean(torch.abs(torch.diff(z,dim=1)),dim=(0,1))
    #speed=soft_median(torch.abs(torch.diff(z,dim=1)))

    #speed,_ = soft_median(torch.abs(torch.diff(z,dim=1)).flatten(start_dim=0, end_dim=1),dim=0)
    speed_penalty = torch.mean(speed)

    # Persistence penalty
    curr_projection = densMAPprojector(z.reshape(-1,z.shape[2]).t())
    if epoch == 0:
        initial_projection = curr_projection.detach()
    persistence_penalty = nn.MSELoss()(curr_projection, initial_projection)

    # Response penalty
    x_reshaped = x.reshape(-1, x.shape[2]).T
    x_norm = (x_reshaped - torch.mean(x_reshaped, dim=1, keepdim=True)) / (torch.std(x_reshaped, dim=1, keepdim=True) + 1e-8)
    #response_penalty = nn.MSELoss()(torch.FloatTensor(X_rnn.detach()[speed_rnn_original<threshold,:]), x_norm[speed_rnn_original<threshold,:])

    #Curvature penalty
   # curvature = get_curvature(net_contract,u.unsqueeze(1),dt,steps)
    #curvature_penalty = torch.mean((curvature-curvature_target[None,:,None])**2)
    # Store projection and speeds


    # Calculate loss and backpropagate
    loss =  0*variance_penalty + 0*activity_penalty + 1 * speed_penalty + 0.1 * persistence_penalty   #+ response_penalty#+ 0*curvature_penalty
    #loss =  activity_penalty
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f} | variance = {variance_penalty.item():.4f} | active = {activity_penalty.item():.4f} | speed = {speed_penalty.item():.4f} | persist = {persistence_penalty.item():.4f}   ")


def get_curvature_tensor(model, u, dt, steps):
    """
    Computes curvature for each neuron at each time step and trial as a torch tensor.
    Curvature = (h_dot_i)^2 * f''(h_i)
    """
    y = model(u, dt, steps)
    input_eff = model.input_layer(u).unsqueeze(1).expand(-1, steps, -1)
    h = model.recurrent_layer(y) + input_eff

    # Compute f''(h) for softplus
    beta = model.beta
    f_prime = torch.sigmoid(beta * h)
    f_double_prime = beta * f_prime * (1 - f_prime)

    # Compute y_dot analytically: y_dot = -alpha*y + alpha*f(h)
    # Note: In contraction.py forward pass:
    # state_new = (1 - self.alpha * dt) * states[:, i, :] + self.alpha * dt * f(h)
    # (state_new - states[:, i, :]) / dt = alpha * (f(h) - states[:, i, :])
    # So y_dot = alpha * (f(h) - y)
    y_dot = model.alpha.unsqueeze(0).unsqueeze(0) * (model.activation(h) - y)

    # Compute h_dot = W * y_dot
    h_dot = torch.matmul(y_dot, model.recurrent_layer.weight.t())

    # Curvature = (h_dot)^2 * f''(h)
    curvature = (h_dot**2) * f_double_prime
    return curvature


projection = densMAPprojector(z.reshape(-1,z.shape[2]).t().detach())
projection_np = projection.detach().cpu().numpy()

plt.figure(figsize=(5, 5))
plt.scatter(projection_np[:, 0], projection_np[:, 1], color='gray', alpha=0.5, s=1)
plt.title("Projection Scatterplot")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig('projection_scatterplot.png', dpi=300, bbox_inches='tight')
plt.close()


# Train net_spike to match net_contract curvature
print("\nTraining net_spike to match net_contract curvature...")
net_spike = torch.load('CP_net.pth', weights_only=False, map_location=device).to(device)
optimizer_spike = optim.Adam(net_spike.parameters(), lr=0.0001)
criterion_spike = nn.MSELoss()

# Target curvature from trained net_contract
with torch.no_grad():
    curvature_target = get_curvature_tensor(net_contract, u.unsqueeze(1), dt, steps)

for epoch in range(10000):
    optimizer_spike.zero_grad()
    curvature_spike = get_curvature_tensor(net_spike, u.unsqueeze(1), dt, steps)
    
    loss_spike = criterion_spike(curvature_spike, curvature_target)
    
    loss_spike.backward()
    optimizer_spike.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: net_spike Loss = {loss_spike.item():.6f}")

print("Training net_spike completed.")

# Plot net_spike projection
with torch.no_grad():
    y_spike = net_spike(u.unsqueeze(1), dt, steps)
    z_spike = (y_spike - torch.mean(y_spike, dim=(0, 1))) / torch.std(y_spike, dim=(0, 1))
    projection_spike = densMAPprojector(z_spike.reshape(-1, z_spike.shape[2]).t().detach())
    projection_spike_np = projection_spike.cpu().numpy()

plt.figure(figsize=(5, 5))
plt.scatter(projection_spike_np[:, 0], projection_spike_np[:, 1], color='red', alpha=0.5, s=1)
plt.title("Net Spike Projection Scatterplot")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig('projection_scatterplot_spike.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved net_spike projection to projection_scatterplot_spike.png")

