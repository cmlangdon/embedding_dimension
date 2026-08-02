# BWMRegionAnalysis_Spatial.ipynb

#%%
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath('../../'))

from BWM.bwm_functions import *
from BWM.models.spatial_isometric_autoencoder import SpatialIsometricAutoencoder

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

#%%
# Set directory path and select an EID
directory_path = '../Data2'
all_items = os.listdir(directory_path)
eids = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]

#%%
# Data loading using get_region_response_matrix
region = 'MRN' # Midbrain reticular nucleus
response_matrix_train, condition_matrix, neuron_eids = get_region_response_matrix(
    eids=eids,
    directory_path=directory_path,
    region=region,
    mapping='Beryl',
    sigma=5,
    n_steps=100,
    threshold=5,
    min_trials_per_condition=50,
    n_pseudo=25,
    conditions=['choice'],
    event="stimOn_times",
    filter_significant=True,
    significance_threshold=0.6
)

# Average over pseudo-trials for fitting
responses = np.mean(response_matrix_train, axis=0) 
# Z-score and reshape
responses_norm = (responses - np.mean(responses, axis=(0, 1), keepdims=True)) / (np.std(responses, axis=(0, 1), keepdims=True) + 1e-8)
response_matrix_flat = responses_norm.reshape(-1, responses_norm.shape[2])

#%%
def calculate_msd(responses):
    """
    Calculate Mean Standardized Derivative using finite differences.
    responses: (conditions, time, neurons)
    """
    # Finite difference along time axis (axis 1)
    diff_responses = np.diff(responses, axis=1)
    # Absolute standardized derivative (already standardized in the previous cell)
    # Mean over conditions and time
    msd = np.mean(np.abs(diff_responses), axis=(0, 1))
    return msd

msd = calculate_msd(responses_norm)

#%%
# Fit Spatial Isometric Flow Model
# Transpose -> (neurons, conditions * time)
X_transposed = response_matrix_flat.T

# Scale each row (neuron) to unit length
row_norms = np.linalg.norm(X_transposed, axis=1, keepdims=True)
X_pre_pca = X_transposed / (row_norms + 1e-8)

# Reduce features to 20 using PCA
pca_model = PCA(n_components=20)
X_input = pca_model.fit_transform(X_pre_pca)
print(f"Input to spatial isometric flow model shape: {X_input.shape}")

# Encode neuron EIDs as labels for spatial decodability
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_eid = le.fit_transform(neuron_eids)

# Initialize and fit spatial autoencoder
autoencoder = SpatialIsometricAutoencoder(
    latent_dim=2, hidden_dim=128, depth=4,
    epochs=4000, batch_size=128,
    lambda_iso=0.1, lambda_decodability=1.0, 
    verbose=True, device=device, patience=100,
    lr=.0001, r2_diff_threshold=0.2, lambda_density=0.1
)
autoencoder.fit(X_input, labels=y_eid)

#%%
# Transform to latent space and plot
z_neurons = autoencoder.transform(X_input)
from BWM.models.isometric_autoencoder import get_density_proxy
log_px_latent = -get_density_proxy(X_input, k=10).numpy()

# Check EID decoding accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
scores = cross_val_score(clf, z_neurons, y_eid, cv=5)
print(f"EID decoding accuracy from latent space: {scores.mean():.3f} +/- {scores.std():.3f}")

# Also test the built-in decoder
y_pred = autoencoder.predict_labels(X_input)
accuracy = np.mean(y_pred == y_eid)
print(f"Built-in decoder accuracy: {accuracy:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

# Subplot 1: Gray
axes[0].scatter(z_neurons[:, 0], z_neurons[:, 1], c='gray', s=2, alpha=0.2)
axes[0].set_title("Encoder Output (Gray)")
axes[0].set_axis_off()
axes[0].set_aspect('equal')

# Subplot 2: Colored by MSD
vmin_msd, vmax_msd = np.percentile(msd, [0, 100])
range_msd = vmax_msd - vmin_msd
vmin_msd -= 0.1 * range_msd
vmax_msd += 0.1 * range_msd
sc1 = axes[1].scatter(z_neurons[:, 0], z_neurons[:, 1], c=msd, cmap='turbo', s=1, alpha=0.4, vmin=vmin_msd, vmax=vmax_msd)
axes[1].set_title(f"Colored by MSD")
axes[1].set_axis_off()
axes[1].set_aspect('equal')

# Subplot 3: Colored by Log Probability
vmin_logp, vmax_logp = np.percentile(log_px_latent, [1, 99])
range_logp = vmax_logp - vmin_logp
vmin_logp -= 0.1 * range_logp
vmax_logp += 0.1 * range_logp
sc2 = axes[2].scatter(z_neurons[:, 0], z_neurons[:, 1], c=log_px_latent, cmap='turbo_r', s=.5, alpha=0.3, vmin=vmin_logp, vmax=vmax_logp)
axes[2].set_title("Colored by Log Prob")
axes[2].set_axis_off()
axes[2].set_aspect('equal')

rho, _ = spearmanr(log_px_latent, msd)
plt.suptitle(f"Spatial Isometric Flow Autoencoder (Region: {region}, rho={rho:.3f})")
sns.despine()
plt.tight_layout()
plt.savefig("../Figures/BWMRegionAnalysis_Spatial_Embedding.png", bbox_inches='tight', dpi=300, transparent=True)
plt.show()

#%%
# densMAP visualization
import umap
from scipy.stats import gaussian_kde

# Fit densMAP
reducer = umap.UMAP(densmap=True, dens_lambda = 2, n_neighbors=2000)
z_densmap = reducer.fit_transform(X_input)

# Calculate KDE for densMAP coordinates
xy = np.vstack([z_densmap[:, 0], z_densmap[:, 1]])
kde = gaussian_kde(xy)(xy)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

# Subplot 1: Gray
axes[0].scatter(z_densmap[:, 0], z_densmap[:, 1], c='gray', s=2, alpha=0.2)
axes[0].set_title("densMAP (Gray)")
axes[0].set_axis_off()
axes[0].set_aspect('equal')

# Subplot 2: Colored by KDE
vmin_kde, vmax_kde = np.percentile(kde, [0, 100])
range_kde = vmax_kde - vmin_kde
vmin_kde -= 0.1 * range_kde
vmax_kde += 0.1 * range_kde
axes[1].scatter(z_densmap[:, 0], z_densmap[:, 1], c=kde, cmap='turbo_r', s=1, alpha=0.4, vmin=vmin_kde, vmax=vmax_kde)
axes[1].set_title("Colored by KDE")
axes[1].set_axis_off()
axes[1].set_aspect('equal')

# Subplot 3: Colored by MSD
vmin_msd_dm, vmax_msd_dm = np.percentile(msd, [0, 100])
range_msd_dm = vmax_msd_dm - vmin_msd_dm
vmin_msd_dm -= 0.1 * range_msd_dm
vmax_msd_dm += 0.1 * range_msd_dm
axes[2].scatter(z_densmap[:, 0], z_densmap[:, 1], c=msd, cmap='turbo', s=1, alpha=0.4, vmin=vmin_msd_dm, vmax=vmax_msd_dm)
axes[2].set_title("Colored by MSD")
axes[2].set_axis_off()
axes[2].set_aspect('equal')

plt.suptitle(f"densMAP Projection (Region: {region})")
sns.despine()
plt.tight_layout()
plt.savefig("../Figures/BWMRegionAnalysis_Spatial_densMAP.png", bbox_inches='tight', dpi=300, transparent=True)
plt.show()

#%%
