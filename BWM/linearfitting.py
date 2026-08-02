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
        x = torch.FloatTensor(x)

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
    response_tensor_norm = (response_tensor - np.mean(response_tensor, axis=(0, 1), keepdims=True)) / (
                np.std(response_tensor, axis=(0, 1), keepdims=True) + 1e-8)
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
    response_tensor_norm = (response_tensor - np.mean(response_tensor, axis=(0, 1), keepdims=True)) / (
                np.std(response_tensor, axis=(0, 1), keepdims=True) + 1e-8)
    diff_responses = np.diff(response_tensor_norm, axis=1)
    # Absolute standardized derivative
    msd = np.mean(np.abs(diff_responses), axis=(0, 1))
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
    y_dot = model.alpha * (-y + model.activation(h))

    # Compute h_dot analytically
    h_dot = torch.matmul(y_dot, model.recurrent_layer.weight.t())

    # Compute curvature = (h_dot)^2 * f''(h)
    curvature = (h_dot ** 2) * f_double_prime

    return curvature.detach().cpu().numpy(), f_double_prime.detach().cpu().numpy()


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
        self.alpha = .05

    def forward(self, u, dt, steps):
        batch_size = u.shape[0]
        states = torch.zeros(batch_size, 1, self.n, device=device)
        for i in range(steps - 1):
            state_new = (1 - self.alpha * dt) * states[:, i, :] + self.alpha * dt * (
                self.recurrent_layer(states[:, i, :]) + self.input_layer(u))
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        return states


def main():
    # Randomly choose parameters
    bias = 2
    beta = 2
    dt = 1
    active_lambda = .1

    # directory_path = 'BWM/Data2'
    # if not os.path.exists(directory_path):
    #     print(f"Directory {directory_path} not found. Please ensure you are running from the project root.")
    #     return
    #
    # all_items = os.listdir(directory_path)
    # eids = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]
    #
    # # Choose a random Beryl region
    # from iblatlas.regions import BrainRegions
    # br = BrainRegions()
    # print("Identifying available Beryl regions...")
    # available_regions = set()
    # # Sample some EIDs to find regions quickly
    # for eid in random.sample(eids, min(50, len(eids))):
    #     cluster_file = os.path.join(directory_path, eid, 'clusters.pkl')
    #     if os.path.exists(cluster_file):
    #         try:
    #             clusters = pd.read_pickle(cluster_file)
    #             if 'atlas_id' in clusters.columns:
    #                 mapped = br.id2acronym(clusters['atlas_id'], mapping='Beryl')
    #                 available_regions.update(mapped)
    #         except Exception:
    #             continue
    #
    # # Filter out 'root', 'void', 'x', 'y' or any other non-brain regions if necessary
    # available_regions = [r for r in available_regions if r not in ['root', 'void', 'x', 'y']]
    #
    # if not available_regions:
    #     print("No Beryl regions found. Defaulting to 'CP'.")
    #     region = 'CP'
    # else:
    #     region = random.choice(available_regions)

    # region = 'CP'
    #
    # print(f"Randomly chosen region: {region}")
    # print(f"Randomly chosen parameters: bias={bias:.4f}, beta={beta:.4f}, dt={dt:.4f}, active_lambda={active_lambda:.4f}")
    #
    # print(f"Loading data for region {region}...")
    # pseudo_response_tensor, condition_matrix, neuron_eids = get_region_response_matrix(
    #     eids=eids,
    #     directory_path=directory_path,
    #     region=region,
    #     mapping='Beryl',
    #     sigma=2,
    #     n_steps=100,
    #     threshold=1,
    #     min_trials_per_condition=50,
    #     n_pseudo=25,
    #     conditions=['choice'],
    #     event="stimOn_times",
    #     filter_significant=False,
    #     significance_threshold=0.
    # )
    #
    # if pseudo_response_tensor is None or pseudo_response_tensor.size == 0:
    #     print(f"No significant data found for region {region}. Exiting.")
    #     return
    #
    # n_neurons = pseudo_response_tensor.shape[3]
    # if n_neurons <= 1000:
    #     print(f"Region {region} has only {n_neurons} neurons, which is not > 1000. Exiting.")
    #     return

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

    print("Training RNN...")
    for epoch in range(5000):
        optimizer.zero_grad()
        x = net(u.unsqueeze(1), dt, steps)
        x_reshaped = x.reshape(-1, x.shape[2]).T
        activity_penalty = torch.std(torch.mean(x_reshaped, dim=1)) + torch.std(torch.std(x_reshaped, dim=1))
        x_norm = (x_reshaped - torch.mean(x_reshaped, dim=1, keepdim=True)) / (
                    torch.std(x_reshaped, dim=1, keepdim=True) + 1e-8)

        loss = criterion(torch.FloatTensor(X_input).to(device), x_norm) + active_lambda * activity_penalty
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    print("Generating summary plot...")
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 3)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    ax6 = plt.subplot(gs[1, 2])
    ax7 = plt.subplot(gs[2, 0])
    ax8 = plt.subplot(gs[2, 1])
    ax9 = plt.subplot(gs[2, 2])

    # First row: Data
    z_neurons_data = umap.UMAP(densmap=True, dens_lambda=2, n_neighbors=min(500, X_input.shape[0] - 1)).fit_transform(
        X_input)
    loc_density_data = get_density_proxy(torch.FloatTensor(X_input), k=min(50, X_input.shape[0] - 1)).numpy()

    sns.scatterplot(x=z_neurons_data[:, 0], y=z_neurons_data[:, 1], color='gray', s=2, alpha=.35, ax=ax1)
    sns.scatterplot(x=z_neurons_data[:, 0], y=z_neurons_data[:, 1], c=loc_density_data, cmap='turbo', s=2, alpha=.5,
                    ax=ax2)

    vmin_msd_data, vmax_msd_data = np.percentile(msd_data, [1, 99])
    sns.scatterplot(x=z_neurons_data[:, 0], y=z_neurons_data[:, 1], c=msd_data, cmap='turbo', s=2, alpha=.5, ax=ax3)

    # Second row: RNN
    X_rnn = net(u.unsqueeze(1), dt, 100)
    X_rnn_norm = (X_rnn - torch.mean(X_rnn, dim=(0, 1), keepdim=True)) / (
                torch.std(X_rnn, dim=(0, 1), keepdim=True) + 1e-8)
    msd_rnn = torch.mean(torch.abs(torch.diff(X_rnn_norm, dim=1)), dim=(0, 1)).detach().cpu().numpy()
    X_rnn_final = X_rnn_norm.reshape(-1, X_rnn_norm.shape[2]).T.detach().cpu().numpy()

    z_neurons_rnn = umap.UMAP(densmap=True, dens_lambda=2,
                              n_neighbors=min(500, X_rnn_final.shape[0] - 1)).fit_transform(X_rnn_final)
    loc_density_rnn = get_density_proxy(torch.FloatTensor(X_rnn_final), k=min(50, X_rnn_final.shape[0] - 1)).numpy()

    sns.scatterplot(x=z_neurons_rnn[:, 0], y=z_neurons_rnn[:, 1], color='gray', s=2, alpha=.35, ax=ax4)
    sns.scatterplot(x=z_neurons_rnn[:, 0], y=z_neurons_rnn[:, 1], c=loc_density_rnn, cmap='turbo', s=2, alpha=.5,
                    ax=ax5)

    vmin_msd_rnn, vmax_msd_rnn = np.percentile(msd_rnn, [1, 99])
    sns.scatterplot(x=z_neurons_rnn[:, 0], y=z_neurons_rnn[:, 1], c=msd_rnn, cmap='turbo', s=2, alpha=.5, ax=ax6)

    # Third row: Correlations and Curvature
    sns.scatterplot(x=loc_density_data, y=msd_data, ax=ax7, s=3, alpha=.5)
    ax7.set_xlabel('N.N. Distance (Data)')
    ax7.set_ylabel('MSD (Data)')

    sns.scatterplot(x=loc_density_rnn, y=msd_rnn, ax=ax8, s=3, alpha=.5)
    ax8.set_xlabel('N.N. Distance (RNN)')
    ax8.set_ylabel('MSD (RNN)')

    n_groups = 50
    msd_sorted_indices = np.argsort(msd_rnn)
    group_size = n_neurons // n_groups
    groups = [msd_sorted_indices[i * group_size:(i + 1) * group_size] for i in range(n_groups)]
    if n_neurons % n_groups != 0:
        groups[-1] = np.concatenate([groups[-1], msd_sorted_indices[n_groups * group_size:]])

    curvature, _ = get_curvature(net, u.unsqueeze(1), .25, steps)
    cmap = plt.get_cmap('cividis')
    colors = [cmap(i / (n_groups - 1)) for i in range(n_groups)] if n_groups > 1 else [cmap(0.5)]

    for g_idx, group in enumerate(groups):
        color = colors[g_idx]
        group_mean = np.mean(curvature[:, :, group], axis=(0, 2))
        ax9.plot(group_mean, color=color, alpha=1.0, lw=1, label=f'G{g_idx + 1}')
    ax9.set_xlabel('Time steps')
    ax9.set_ylabel('Curvature')

    # Title with parameters
    fig.suptitle(
        f"Density Matching Summary: {region} (N={n_neurons})\n(bias={bias:.2f}, beta={beta:.2f}, dt={dt:.2f}, lambda={active_lambda:.2f})",
        fontsize=12)

    # Cleanup axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    output_filename = "density_matching_summary" + f"_{region}.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_filename}")
    torch.save(net, 'CP_linearnet.pth')


if __name__ == "__main__":
    main()
