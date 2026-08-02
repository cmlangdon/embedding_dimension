import pandas as pd
import scipy.stats
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import umap
from sklearn.decomposition import PCA
import torch
import os
import sys
import torch.optim as optim
from bwm_functions import *

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set plotting parameters
plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .5
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

    if batch_size > 2000:
        avg_dists = []
        for i in range(0, batch_size, 1000):
            batch_x = x[i:i + 1000]
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
        # u is expected to be (batch_size, input_dim)
        batch_size = u.shape[0]
        states = torch.zeros(batch_size, 1, self.n, device=device)
        for i in range(steps - 1):
            # recurrent_layer(states[:, i, :]) is (batch_size, n)
            # input_layer(u) is (batch_size, n)
            if self.non_linear:
                state_new = (1 - self.alpha * dt) * states[:, i, :] + self.alpha * dt * (
                    self.activation(self.recurrent_layer(states[:, i, :]) + self.input_layer(u)))
            else:
                state_new = (1 - self.alpha * dt) * states[:, i, :] + self.alpha * dt * (
                    self.recurrent_layer(states[:, i, :]) + self.input_layer(u))
            states = torch.cat((states, state_new.unsqueeze(1)), 1)
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
    # Parameters
    bias = 2
    beta = 2
    dt = 1
    active_lambda = .1
    lr = .0002
    steps = 100
    top_k = 100
    n_null_models = 100

    # Load and preprocess CP response data
    print("Loading CP response data...")
    # Adjust path if necessary. Script is in BWM/lineartest.py
    data_path = 'ResponseData/CP_pseudo_responses.npy'
    if not os.path.exists(data_path):
        data_path = 'BWM/ResponseData/CP_pseudo_responses.npy'
    
    pseudo_response_tensor = np.load(data_path)
    X_input = iso_ae_preprocessing(pseudo_response_tensor)
    n_neurons = pseudo_response_tensor.shape[3]
    
    # Compute observed metrics directly on CP data
    print("Computing observed metrics on CP data...")
    X_input_torch = torch.from_numpy(X_input).float().to(device)
    
    # Target FVE for linear RNNs (dimensionality matching)
    target_fve = efficient_variance_explained_svd_vals(X_input_torch, top_k=top_k).detach()
    u = torch.FloatTensor(np.linspace(-1, 1, 2)).to(device)
    criterion = nn.MSELoss()

    # Manifold and Density for CP data
    # CP data in X_input is (n_neurons, kt)
    z_neurons_cp = umap.UMAP(densmap=True, dens_lambda=2, n_neighbors=min(500, X_input.shape[0]-1)).fit_transform(X_input)
    loc_density_cp = get_density_proxy(X_input, k=min(50, X_input.shape[0]-1)).cpu().numpy()
    
    # Reshape CP data back to (kt, n_neurons) for activity-based metrics
    # X_input is (n_neurons, kt)
    X_cp_activity = X_input.T
    
    # Calculate MSD for CP data
    # pseudo_response_tensor is (n_trials, n_neurons, n_time_points)
    # Average across trials for MSD consistency
    response_tensor = np.mean(pseudo_response_tensor, axis=0) # (2, 100, n_neurons) - Wait, shape is (128, 2, 100, 11270)
    # If axis=0, then response_tensor is (2, 100, 11270)
    # The previous code in ManifoldAnalysis.ipynb:
    # response_tensor = np.mean(pseudo_response_tensor, axis=0) # (2, 100, 11270)
    # response_tensor_norm = (response_tensor - np.mean(response_tensor, axis=(0, 1), keepdims=True)) / (np.std(response_tensor, axis=(0, 1), keepdims=True) + 1e-8)
    # msd_cp = np.mean(np.abs(np.diff(response_tensor_norm, axis=1)), axis=(0, 1)) # (11270,)
    
    response_tensor = np.mean(pseudo_response_tensor, axis=0)
    response_tensor_norm = (response_tensor - np.mean(response_tensor, axis=(0, 1), keepdims=True)) / (np.std(response_tensor, axis=(0, 1), keepdims=True) + 1e-8)
    msd_cp = np.mean(np.abs(np.diff(response_tensor_norm, axis=1)), axis=(0, 1)) # Result should be (n_neurons,)
    
    corr_observed, _ = scipy.stats.spearmanr(loc_density_cp, msd_cp)
    
    avg_manifold_density = np.mean(loc_density_cp)
    ldi_observed = calculate_ldi(X_cp_activity, loc_density_cp, avg_manifold_density)

    print(f"Observed CP Corr(Density, MSD): {corr_observed:.4f}")
    print(f"Observed CP LDI: {ldi_observed:.4f}")

    # Significance Testing with Linear RNNs
    print(f"Training {n_null_models} linear RNNs for significance testing...")
    null_corrs = []
    null_ldis = []
    
    for i in range(n_null_models):
        print(i)
        linear_net = RNN(n=n_neurons, input_dim=1, beta=beta, bias=bias, non_linear=False).to(device)
        linear_optimizer = optim.Adam(linear_net.parameters(), lr=lr)
        
        for epoch in range(1000):
            linear_optimizer.zero_grad()
            x_lin = linear_net(u.unsqueeze(1), dt, steps) # u is (2,), u.unsqueeze(1) is (2, 1)
            x_lin_reshaped = x_lin.reshape(-1, x_lin.shape[2]).T
            x_lin_norm = (x_lin_reshaped - torch.mean(x_lin_reshaped, dim=1, keepdim=True)) / (torch.std(x_lin_reshaped, dim=1, keepdim=True) + 1e-8)
            
            fve_lin = efficient_variance_explained_svd_vals(x_lin_norm.t(), top_k=top_k)
            loss_lin = criterion(fve_lin, target_fve)
            loss_lin.backward()
            linear_optimizer.step()
            
        with torch.no_grad():
            x_lin = linear_net(u.unsqueeze(1), dt, 100)
            # x_lin is (2, 100, n_neurons)
            x_lin_norm = (x_lin - torch.mean(x_lin, dim=(0, 1), keepdim=True)) / (torch.std(x_lin, dim=(0, 1), keepdim=True) + 1e-8)
            x_lin_final = x_lin_norm.reshape(-1, x_lin_norm.shape[2]).T
            
            loc_density_lin = get_density_proxy(x_lin_final, k=min(50, x_lin_final.shape[0]-1)).cpu().numpy()
            msd_lin = torch.mean(torch.abs(torch.diff(x_lin_norm, dim=1)), dim=(0, 1)).cpu().numpy()
            
            c_null, _ = scipy.stats.spearmanr(loc_density_lin, msd_lin)
            null_corrs.append(c_null)
            
            act_lin = x_lin_norm.reshape(-1, x_lin_norm.shape[2]).cpu().numpy()
            null_ldis.append(calculate_ldi(act_lin, loc_density_lin, np.mean(loc_density_lin)))
        
        if (i+1) % 10 == 0:
            print(f"Linear RNN {i+1}/{n_null_models} trained.")

    p_corr = np.mean(np.array(null_corrs) >= corr_observed)
    p_ldi = np.mean(np.array(null_ldis) >= ldi_observed)

    # Visualization
    print("Generating results figure...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 1. Density Map
    sc1 = axes[0].scatter(z_neurons_cp[:, 0], z_neurons_cp[:, 1], c=loc_density_cp, cmap='turbo', s=2, alpha=.5)
    plt.colorbar(sc1, ax=axes[0], label='Density')
    axes[0].set_title('CP DensMAP: Density')
    
    # 2. MSD Map
    sc2 = axes[1].scatter(z_neurons_cp[:, 0], z_neurons_cp[:, 1], c=msd_cp, cmap='turbo', s=2, alpha=.5)
    plt.colorbar(sc2, ax=axes[1], label='MSD')
    axes[1].set_title('CP DensMAP: MSD')
    
    # 3. MSD-Density Corr Null Distribution
    axes[2].hist(null_corrs, bins=20, color='gray', alpha=0.7)
    axes[2].axvline(corr_observed, color='red', linestyle='--', label=f'Obs: {corr_observed:.3f}')
    axes[2].set_title(f'Spearman Corr Null\n(p={p_corr:.4f})')
    axes[2].set_xlabel('Correlation')
    axes[2].legend()

    # 4. LDI Null Distribution
    axes[3].hist(null_ldis, bins=20, color='blue', alpha=0.7)
    axes[3].axvline(ldi_observed, color='red', linestyle='--', label=f'Obs: {ldi_observed:.3f}')
    axes[3].set_title(f'LDI Null\n(p={p_ldi:.4f})')
    axes[3].set_xlabel('LDI')
    axes[3].legend()

    plt.tight_layout()
    os.makedirs('BWM/Figures', exist_ok=True)
    save_path = 'BWM/Figures/linear_test_significance.png'
    plt.savefig(save_path, dpi=300)
    print(f"Results figure saved to {save_path}")

if __name__ == "__main__":
    main()
