import pandas as pd
import skdim
import scipy

from scipy.stats import gaussian_kde
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import random
import string
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from iblatlas.regions import BrainRegions
br = BrainRegions()
import umap
from iblatlas.regions import BrainRegions
br = BrainRegions()
from bwm_functions import *
import os
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
import torch
import numpy as np
plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .5



from scipy.spatial import KDTree
from scipy.stats import norm
def clark_evans_with_pvalue(points, area):
    n = len(points)
    num_dims = points.shape[1]
    density = n / area

    # 1. Nearest Neighbor distances
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)
    r_obs = np.mean(distances[:, 1])

    # 2. Expected distance (2D specific for this SE formula)
    r_exp = 1 / (2 * np.sqrt(density))

    # 3. Standard Error (for 2D)
    se_r = np.sqrt((4 - np.pi) / (4 * np.pi * n * density))

    # 4. Z-score and P-value (Two-tailed)
    z_score = (r_obs - r_exp) / se_r
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

    R = r_obs / r_exp

    return R, z_score, p_value




import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# 1. Generate sample data


def find_best_k(data, max_k=10):
    scores = []
    k_range = range(2, max_k + 1) # Silhouette score requires at least 2 clusters

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
        print(f"Clusters: {k}, Silhouette Score: {score:.4f}")

    # Find the maximum score and its corresponding K
    max_score = max(scores)
    best_k = k_range[scores.index(max_score)]

    return best_k, max_score, scores




# Loop through data and calculate FID and its correlation with density. Restrict to recordings with >500 neurons.

directory_path = 'Data2'
all_items = os.listdir(directory_path)
print(len(all_items))
# Filter the list to include only folders (directories)
eids = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]
#eids = np.load("eids.npy",allow_pickle=True)
FIDS = []
Silhouettes = []
Clark = []
Clark_p = []
for eid in eids:

    # eid = random.choice(eids)
    full_path_to_random_folder = os.path.join(directory_path, eid)
    print(f"Randomly selected eid: {full_path_to_random_folder}")

    # for _ in range(1):

    dim = 5
    sample_size = 50
    sigma = 5
    n_steps = 100
    threshold = .01
    min_trials_per_condition = 50
    conditions = ['choice']
    event = "stimOn_times"
    clusters = pd.read_pickle(full_path_to_random_folder + '/clusters.pkl')

    clusters['cosmos'] = br.id2acronym(clusters['atlas_id'], mapping="Cosmos")

    spikes = pd.read_pickle(full_path_to_random_folder + '/spikes.pkl')
    trials = pd.read_pickle(full_path_to_random_folder + '/trials.pkl')

    counts_series = clusters.groupby('cosmos').size()
    print(counts_series)

    # shape n_pseudo x n_conditions x time x n
    response_matrix_train, condition_matrix_train = get_aligned_response_matrix(
    spikes, clusters, trials, event, conditions,
    stride=.005,
    step=.005,
    n_steps=n_steps,
    sigma=sigma,
    threshold=threshold,
    min_trials_per_condition=min_trials_per_condition,regions = counts_series.index.values)

    # Restrict to active units
    restriction = np.logical_and(np.std(response_matrix_train, axis=(0, 1, 2)) > .01,
                             np.std(response_matrix_train, axis=(0, 1, 2)) > .01)
    response_matrix_train = response_matrix_train[
    :, :, :, restriction]

    z = np.mean(response_matrix_train, axis = 0)
    responses = (response_matrix_train - np.mean(z, axis=(0, 1),
                                                         keepdims=True)) / np.std(z,
                                                                                  axis=(0, 1),
                                                                                  keepdims=True)
    #data_snr =np.mean(np.std(responses, axis=0), axis=(0, 1))
    #data =np.log(np.mean(response_matrix_train, axis=(0, 1,2)))
    responses = np.mean(responses, axis=0)
    response_matrix = responses.reshape(-1, responses.shape[2])


    # Calculate FID

    fisher = skdim.id.FisherS().fit(response_matrix.T)
    id_fisher = fisher.dimension_
    print(f"FisherS Estimator: {id_fisher:.2f}")
    FIDS.append(id_fisher)

    # Calculate correlation between local FID and local density.
    x_proj = PCA(n_components = 2).fit_transform(response_matrix.T)
    kde = gaussian_kde(x_proj.T)

    densities = kde.evaluate(x_proj.T)

    FID = skdim.id.FisherS().fit_pw(response_matrix.T, n_neighbors=25)

    a = FID.dimension_pw_
    b = densities

    mask = ~np.isnan(a) & ~np.isnan(b)

    # Apply the mask to filter both arrays
    a_clean = a[mask]
    b_clean = b[mask]
    print(f"Correlation: {scipy.stats.pearsonr(a_clean, b_clean)[0]:.2f}")

    # Uniformity

    R, z, p = clark_evans_with_pvalue(x_proj, 100**2)
    Clark.append(R)
    Clark_p.append(p)
    print(f"R-Ratio: {R:.4f}")
    print(f"Z-score: {z:.4f}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print("Statistically Significant: Reject the null hypothesis of uniformity.")
    else:
        print("Not Significant: Fail to reject the null hypothesis (Distribution is likely uniform).")

    # Clustering.
    best_k, max_val, all_scores = find_best_k(x_proj)
    Silhouettes.append(max_val)
    print(f"\nOptimal number of clusters: {best_k}")
    print(f"Maximum Silhouette Score: {max_val:.4f}")


np.save("Silhouettes.npy", Silhouettes)
np.save("FIDS.npy", FIDS)
np.save("Clark.npy", Clark)
np.save("Clark_p.npy", Clark_p)

