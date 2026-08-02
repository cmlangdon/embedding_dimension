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
import random

# Add project root to path
# Assuming the script is run from the project root
sys.path.append(os.getcwd())


import torch.optim as optim
from bwm_functions import *

plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .5

# Set global plotting parameters for font size
tick_fontsize = 6
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': tick_fontsize,
    'ytick.labelsize': tick_fontsize,
    'legend.fontsize': 7,
    'figure.titlesize': 7
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_density_proxy(x, k=10):
    """
    Computes the average k-NN distance in the input space as a proxy for inverse density.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().to(device)
    else:
        x = x.to(device)

    batch_size = x.shape[0]
    k = min(k, batch_size - 1)

    # We might need to do this in batches if x is very large
    if batch_size > 2000:
        avg_dists = []
        for i in range(0, batch_size, 1000):
            batch_x = x[i:i + 1000]
            # Compute distances from batch to all points
            dists = torch.cdist(batch_x, x)
            knn_dist, _ = torch.topk(dists, k + 1, largest=False)
            avg_dists.append(knn_dist[:, 1:].mean(dim=1))
        return torch.cat(avg_dists, dim=0)
    else:
        x_dist = torch.cdist(x, x)
        x_knn_dist, _ = torch.topk(x_dist, k + 1, largest=False)
        return x_knn_dist[:, 1:].mean(dim=1)

def iso_ae_preprocessing(pseudo_response_tensor):
    """
    Preprocess neural responses intro response matrix (n x kt) for autoencoder.
    """
    response_tensor = np.mean(pseudo_response_tensor, axis=0)
    # Z-score and reshape
    response_tensor_norm = (response_tensor - np.mean(response_tensor, axis=(0, 1), keepdims=True)) / (np.std(response_tensor, axis=(0, 1), keepdims=True) + 1e-8)
    response_matrix = response_tensor_norm.reshape(-1, response_tensor_norm.shape[2])
    X_transposed = response_matrix.T
    # Scale each row (neuron) to unit length
    row_norms = np.linalg.norm(X_transposed, axis=1, keepdims=True)
    X_pre_pca = X_transposed / (row_norms + 1e-8)
    return X_pre_pca

def calculate_msd_data(response_tensor):
    """
    Calculate Mean Standardized Derivative using finite differences.
    responses: (conditions, time, neurons)
    """
    response_tensor_norm = (response_tensor - np.mean(response_tensor, axis=(0, 1), keepdims=True)) / (np.std(response_tensor, axis=(0, 1), keepdims=True) + 1e-8)
    diff_responses = np.diff(response_tensor_norm, axis=1)
    # Absolute standardized derivative
    msd = np.mean(np.abs(diff_responses), axis=(0,1))
    return msd

def get_curvature(model, u, dt, steps):
    """
    Computes curvature for each neuron at each time step and trial.
    Curvature = (h_dot_i)^2 * f''(h_i)
    where h_i is the pre-activation input to neuron i.
    """
    y = model(u, dt, steps)
    input_eff = model.input_layer(u).unsqueeze(1).expand(-1, steps, -1)
    h = model.recurrent_layer(y) + input_eff

    # Compute f''(h) for softplus
    beta = model.beta
    f_prime = torch.sigmoid(beta * h)
    f_double_prime = beta * f_prime * (1 - f_prime)

    # Compute y_dot analytically
    y_dot = model.alpha *(-y + model.activation(h))

    # Compute h_dot analytically
    h_dot = torch.matmul(y_dot, model.recurrent_layer.weight.t())

    # Compute curvature = (h_dot)^2 * f''(h)
    curvature = (h_dot**2) * f_double_prime

    return curvature.detach().cpu().numpy(), f_double_prime.detach().cpu().numpy()

class RNN(torch.nn.Module):
    def __init__(self, n, input_dim, beta, bias, non_linear=True):
        super(RNN, self).__init__()
        self.n = n
        self.input_dim = input_dim
        self.beta = beta
        self.bias = bias
        self.non_linear = non_linear
        self.activation = torch.nn.Softplus(beta=self.beta)
        self.recurrent_layer = nn.Linear(self.n, self.n, bias=True)
        self.recurrent_layer.bias.data.normal_(mean=self.bias, std=0)
        self.recurrent_layer.bias.requires_grad = True
        self.input_layer = nn.Linear(self.input_dim, self.n, bias=False)
        self.alpha = .05
    def forward(self, u, dt, steps):
        batch_size = u.shape[0]
        states = torch.zeros(batch_size, 1, self.n, device=device)
        for i in range(steps - 1):
            if self.non_linear:
                state_new = (1 - self.alpha * dt) * states[:, i, :] + self.alpha *dt * (
                    self.activation(self.recurrent_layer(states[:, i, :]) + self.input_layer(u)))
            else:
                state_new = (1 - self.alpha * dt) * states[:, i, :] + self.alpha *dt * (
                    self.recurrent_layer(states[:, i, :]) + self.input_layer(u))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states

def calculate_ldi(activity, densities, avg_manifold_density):
    # activity: (n_points, n_neurons)
    # densities: (n_neurons,)
    weighted_densities = np.sum(activity * densities, axis=1) / (np.sum(activity, axis=1) + 1e-8)
    awmd = np.mean(weighted_densities)
    return awmd / (avg_manifold_density + 1e-8)

def efficient_variance_explained_svd_vals(responses, top_k):
    x_centered = responses - responses.mean(dim=0, keepdim=True)
    S = torch.linalg.svdvals(x_centered)
    eigenvalues = S**2
    actual_k = min(top_k, len(eigenvalues))
    return eigenvalues[:actual_k] / (eigenvalues.sum() + 1e-8)

def main():
    # Randomly choose parameters
    bias = 2
    beta = 2
    dt = 1
    active_lambda = .1



    region = 'CP'

    pseudo_response_tensor = np.load('ResponseData/CP_pseudo_responses.npy')
    X_input = iso_ae_preprocessing(pseudo_response_tensor)
    msd_data = calculate_msd_data(np.mean(pseudo_response_tensor, axis=0))
    lr = .0002
    steps = 100
    n_neurons = pseudo_response_tensor.shape[3]
    net = RNN(n=n_neurons, input_dim=1, beta=beta, bias=bias).to(device)
    u = torch.FloatTensor(np.linspace(-1, 1, 2)).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    top_k = 100
    target_fve = efficient_variance_explained_svd_vals(torch.from_numpy(X_input).float().to(device), top_k=top_k).detach()

    print("Training original non-linear RNN...")
    for epoch in range(2000):
        optimizer.zero_grad()
        x = net(u.unsqueeze(1), dt, steps)
        x_reshaped = x.reshape(-1, x.shape[2]).T
        activity_penalty = torch.std(torch.mean(x_reshaped, dim=1)) + torch.std(torch.std(x_reshaped, dim=1))
        x_norm = (x_reshaped - torch.mean(x_reshaped, dim=1, keepdim=True)) / (torch.std(x_reshaped, dim=1, keepdim=True) + 1e-8)
        
        loss = criterion(torch.from_numpy(X_input).float().to(device), x_norm) + active_lambda * activity_penalty
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Compute target metrics for the original RNN
    X_rnn = net(u.unsqueeze(1), dt, 100)
    X_rnn_norm = (X_rnn - torch.mean(X_rnn, dim=(0, 1), keepdim=True)) / (torch.std(X_rnn, dim=(0, 1), keepdim=True) + 1e-8)
    X_rnn_final = X_rnn_norm.reshape(-1, X_rnn_norm.shape[2]).T.detach().cpu().numpy()
    
    # Target dimensionality (FVE distribution) for matching
    rnn_fve_target = efficient_variance_explained_svd_vals(torch.from_numpy(X_rnn_final).float().to(device).T, top_k=top_k).detach()

    # Original metrics
    z_neurons_rnn = umap.UMAP(densmap=True, dens_lambda=2, n_neighbors=min(500, X_rnn_final.shape[0]-1)).fit_transform(X_rnn_final)
    loc_density_rnn = get_density_proxy(X_rnn_final, k=min(50, X_rnn_final.shape[0]-1)).cpu().numpy()
    msd_rnn = torch.mean(torch.abs(torch.diff(X_rnn_norm, dim=1)), dim=(0, 1)).detach().cpu().numpy()
    
    corr_observed, _ = scipy.stats.spearmanr(loc_density_rnn, msd_rnn)
    
    avg_manifold_density = np.mean(loc_density_rnn)
    activity_all_trials = X_rnn_norm.reshape(-1, X_rnn_norm.shape[2]).detach().cpu().numpy()
    ldi_observed = calculate_ldi(activity_all_trials, loc_density_rnn, avg_manifold_density)

    # Train 100 linear RNNs to match dimensionality
    print("Training 100 linear RNNs for significance testing...")
    null_corrs = []
    null_ldis = []
    
    for i in range(100):
        linear_net = RNN(n=n_neurons, input_dim=1, beta=beta, bias=bias, non_linear=False).to(device)
        linear_optimizer = optim.Adam(linear_net.parameters(), lr=lr)
        
        for epoch in range(1000): # Fewer epochs for linear match
            linear_optimizer.zero_grad()
            x_lin = linear_net(u.unsqueeze(1), dt, steps)
            x_lin_reshaped = x_lin.reshape(-1, x_lin.shape[2]).T
            x_lin_norm = (x_lin_reshaped - torch.mean(x_lin_reshaped, dim=1, keepdim=True)) / (torch.std(x_lin_reshaped, dim=1, keepdim=True) + 1e-8)
            
            fve_lin = efficient_variance_explained_svd_vals(x_lin_norm.t(), top_k=top_k)
            loss_lin = criterion(fve_lin, rnn_fve_target)
            loss_lin.backward()
            linear_optimizer.step()
        
        # Compute metrics for null model
        with torch.no_grad():
            x_lin = linear_net(u.unsqueeze(1), dt, 100)
            x_lin_norm = (x_lin - torch.mean(x_lin, dim=(0, 1), keepdim=True)) / (torch.std(x_lin, dim=(0, 1), keepdim=True) + 1e-8)
            x_lin_final = x_lin_norm.reshape(-1, x_lin_norm.shape[2]).T.detach().cpu().numpy()
            
            # For null models, we re-run UMAP and density to see if linear structure alone explains it
            # Or use same manifold? User says "significance of these metrics". 
            # Usually means "compared to a model that only matches dimensionality but lacks non-linearity".
            loc_density_lin = get_density_proxy(x_lin_final, k=min(50, x_lin_final.shape[0]-1)).cpu().numpy()
            msd_lin = torch.mean(torch.abs(torch.diff(x_lin_norm, dim=1)), dim=(0, 1)).detach().cpu().numpy()
            
            c_null, _ = scipy.stats.spearmanr(loc_density_lin, msd_lin)
            null_corrs.append(c_null)
            
            act_lin = x_lin_norm.reshape(-1, x_lin_norm.shape[2]).detach().cpu().numpy()
            null_ldis.append(calculate_ldi(act_lin, loc_density_lin, np.mean(loc_density_lin)))
        
        if (i+1) % 10 == 0:
            print(f"Linear RNN {i+1}/100 trained.")

    # Calculate p-values
    p_corr = np.mean(np.array(null_corrs) >= corr_observed)
    p_ldi = np.mean(np.array(null_ldis) >= ldi_observed)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sc1 = ax1.scatter(z_neurons_rnn[:, 0], z_neurons_rnn[:, 1], c=loc_density_rnn, cmap='turbo', s=2, alpha=.5)
    plt.colorbar(sc1, ax=ax1, label='Density')
    ax1.set_title('DensMAP colored by Density')
    
    sc2 = ax2.scatter(z_neurons_rnn[:, 0], z_neurons_rnn[:, 1], c=msd_rnn, cmap='turbo', s=2, alpha=.5)
    plt.colorbar(sc2, ax=ax2, label='MSD')
    ax2.set_title('DensMAP colored by MSD')
    
    fig.suptitle(f'RNN Density Matching Results\n'
                 f'Corr(Density, MSD) = {corr_observed:.4f} (p={p_corr:.4f})\n'
                 f'LDI = {ldi_observed:.4f} (p={p_ldi:.4f})')
    
    plt.tight_layout()
    os.makedirs('BWM/Figures', exist_ok=True)
    save_path = 'BWM/Figures/density_matching_metrics.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

    torch.save(net, 'CP_net.pth')

if __name__ == "__main__":
    main()
