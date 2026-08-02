
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import random
import string
import os
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from scipy.ndimage import gaussian_filter1d
from iblatlas.regions import BrainRegions
br = BrainRegions()
from scipy.stats import gaussian_kde, pearsonr, wilcoxon, ttest_rel, rankdata
from scipy.sparse import save_npz, load_npz, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import umap

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

def get_spike_counts_in_bins(spike_times, spike_clusters, intervals):
    """
    Return the number of spikes in a sequence of time intervals, for each neuron.
    Optimized version using searchsorted and bincount.
    """
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2
    
    cluster_ids = np.unique(spike_clusters)
    if len(cluster_ids) == 0:
        return np.zeros((0, intervals.shape[0]), dtype=np.uint32), cluster_ids
        
    n_neurons = len(cluster_ids)
    n_intervals = intervals.shape[0]
    
    # Map cluster_ids to 0..n_neurons-1
    cluster_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    # Optimized mapping using numpy
    max_id = cluster_ids.max()
    id_map = np.zeros(max_id + 1, dtype=np.int32)
    for i, cid in enumerate(cluster_ids):
        id_map[cid] = i
    spike_cluster_indices = id_map[spike_clusters]
    
    intervals_idx = np.searchsorted(spike_times, intervals)
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    
    # Check if memory usage could be an issue with full matrix
    # If n_intervals * n_neurons is large, we keep it as is.
    # But often n_intervals is large and spikes are sparse.
    
    for j in range(n_intervals):
        i0, i1 = intervals_idx[j, :]
        if i1 > i0:
            counts[:, j] = np.bincount(spike_cluster_indices[i0:i1], minlength=n_neurons)
            
    return counts, cluster_ids


def filter_significant_neurons(spike_times, spike_clusters, event_times, pre_window=[-0.5, 0], post_window=[0, 0.5], threshold=0.6):
    """
    Filter neurons by whether they show a statistically significant response to an event.
    Uses AUC (Area Under the ROC curve) to compare pre and post event spike counts across trials.
    AUC ranges from 0 to 1, where 0.5 is chance.
    """
    n_trials = len(event_times)
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    
    pre_intervals = np.zeros((n_trials, 2))
    post_intervals = np.zeros((n_trials, 2))
    
    for i, t in enumerate(event_times):
        pre_intervals[i] = [t + pre_window[0], t + pre_window[1]]
        post_intervals[i] = [t + post_window[0], t + post_window[1]]
        
    pre_counts, _ = get_spike_counts_in_bins(spike_times, spike_clusters, pre_intervals)
    post_counts, _ = get_spike_counts_in_bins(spike_times, spike_clusters, post_intervals)
    
    # counts shape is (n_neurons, n_trials)
    significant_neurons = []
    
    for i in range(n_neurons):
        pre = pre_counts[i, :]
        post = post_counts[i, :]
        
        if np.all(pre == post):
            continue
            
        # Calculate AUC as a 0-1 statistic
        # AUC = (MeanRank(post) - n_post*(n_post+1)/2) / (n_post * n_pre)
        # We use Mann-Whitney U for this. Since it's paired, we can also use Wilcoxon, 
        # but AUC is traditionally from Mann-Whitney. 
        # rankdata(method='average') is essential to handle ties correctly.
        combined = np.concatenate([pre, post])
        ranks = rankdata(combined, method='average')
        rank_post = np.sum(ranks[len(pre):])
        n1 = len(pre)
        n2 = len(post)
        u2 = rank_post - n2 * (n2 + 1) / 2
        auc = u2 / (n1 * n2)
        
        # We want neurons where response is different from baseline (either higher or lower)
        # so we take absolute deviation from 0.5
        if np.abs(auc - 0.5) + 0.5 > threshold:
            significant_neurons.append(cluster_ids[i])
            
    return np.array(significant_neurons)




# Functions for computing dimensionality and clustering
def compute_pca_error(responses, dim):
    pca = PCA().fit(responses)
    return np.sum(pca.explained_variance_ratio_[dim:])

def compute_kmeans_error(dim, responses):
    kmeans = KMeans(n_clusters=dim).fit(responses)
    responses = responses / np.linalg.norm(responses, axis=1, keepdims=True)
    predicted_responses = np.zeros_like(responses)
    for neuron in range(responses.shape[0]):
        predicted_responses[neuron, :] = kmeans.cluster_centers_[kmeans.labels_[neuron], :]
    predicted_responses = predicted_responses / np.linalg.norm(predicted_responses, axis=1, keepdims=True)
    angles = []
    for neuron in range(responses.shape[0]):
        angles.append(
            np.arccos(np.clip(np.dot(predicted_responses[neuron, :], responses[neuron, :]), -1, 1)) / np.pi * 2)
    return np.mean(angles)


def restrict_to_region(spikes,cluster_df,regions,mapping_col='cosmos'):
    # within_region_cluster_ids = cluster_df[cluster_df.cosmos==cosmos].cluster_id.values
    # all_cluster_ids = spikes['clusters']
   # idx = np.argwhere(cluster_df['cosmos'][spikes['clusters']].isin(cosmos))[:,0]
    
    # Ensure mapping_col exists in cluster_df and index aligns with spike_clusters
    # cluster_df is indexed by cluster_id, but spikes['clusters'] might contain IDs 
    # that are not in cluster_df index if cluster_df is already filtered.
    
    if mapping_col not in cluster_df.columns:
        if mapping_col == 'cosmos':
            cluster_df['cosmos'] = br.id2acronym(cluster_df['atlas_id'], mapping='Cosmos')
        elif mapping_col == 'beryl':
            cluster_df['beryl'] = br.id2acronym(cluster_df['atlas_id'], mapping='Beryl')
        else:
            raise KeyError(f"Column '{mapping_col}' not found in cluster_df and no default mapping available.")

    available_clusters = cluster_df.index
    
    # spikes['clusters'] and spikes['times'] are typically numpy arrays
    spike_clusters_all = np.array(spikes['clusters'])
    spike_times_all = np.array(spikes['times'])
    
    # Print shapes for debugging
    # print(f"DEBUG: spike_clusters_all shape: {spike_clusters_all.shape}")
    # print(f"DEBUG: spike_times_all shape: {spike_times_all.shape}")
    
    if spike_clusters_all.shape[0] != spike_times_all.shape[0]:
        # Handle cases where they might be different lengths due to data corruption or unexpected format
        min_len = min(spike_clusters_all.shape[0], spike_times_all.shape[0])
        spike_clusters_all = spike_clusters_all[:min_len]
        spike_times_all = spike_times_all[:min_len]

    mask = np.isin(spike_clusters_all, available_clusters)
    
    valid_spike_clusters = spike_clusters_all[mask]
    valid_spike_times = spike_times_all[mask]
    
    # Pre-filter spike_times and spike_clusters if regions is not None
    if regions is not None:
        # Use .loc to get region acronyms for valid clusters
        region_acronyms = cluster_df.loc[valid_spike_clusters, mapping_col].values
        region_mask = np.isin(region_acronyms, regions)
        spike_times = valid_spike_times[region_mask]
        spike_clusters = valid_spike_clusters[region_mask]
    else:
        spike_times = valid_spike_times
        spike_clusters = valid_spike_clusters
    
    return spike_clusters, spike_times

def df_2_repsonse_matrix(df, conditions):
    response_matrix = []
    condition_matrix = []
    for pseudo_trial in range(500):
        # Sample with replacement from each condition
        sub_df = df.groupby(conditions).sample(frac=1.0, replace=True)
        # Compute mean response for each condition
        mean_df = sub_df.groupby(conditions).response.apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index()
        # Append
        response_matrix.append(np.stack(mean_df.response.values))
        condition_matrix.append(mean_df[conditions].values)
    response_matrix = np.stack(response_matrix, axis=0)
    condition_matrix = np.stack(condition_matrix, axis=0)
    return response_matrix, condition_matrix


def get_aligned_response_matrix(spikes, clusters, trials, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=5, min_trials_per_condition=10, regions=None, mapping_col='cosmos', n_pseudo=500, return_cluster_ids=False):
    event_times = trials[event].values
    n_trials = len(event_times)

    # Vectorized interval generation
    starts = event_times[:, np.newaxis] - 0.05 + np.arange(n_steps) * step
    intervals = np.stack([starts, starts + stride], axis=2).reshape(-1, 2)

    spike_clusters, spike_times = restrict_to_region(spikes, clusters, regions, mapping_col=mapping_col)
    spike_count, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, intervals=intervals)

    # Reshape spike_counts: (n_neurons, n_trials, n_steps)
    spike_count = spike_count.reshape(len(cluster_ids), n_trials, n_steps)
    # Transpose to (n_trials, n_steps, n_neurons)
    spike_count = spike_count.transpose(1, 2, 0)

    # Vectorized Gaussian filtering
    # Apply to (n_trials, n_steps, n_neurons) along axis 1 (n_steps)
    # Divide by stride to get firing rate
    # Use float32 to save memory
    spike_count = gaussian_filter1d(spike_count.astype(np.float32) / stride, sigma=sigma, axis=1)

    # Pre-calculate contrasts
    contrasts = np.nan_to_num(-trials['contrastLeft'], copy=True, nan=0.0) + \
                np.nan_to_num(trials['contrastRight'], copy=True, nan=0.0)
    
    # Add meta columns to trials for easy grouping
    trials_meta = trials.copy()
    trials_meta['contrast'] = contrasts
    trials_meta['side'] = np.sign(contrasts)
    
    # Filter trials by condition availability
    # conditions can be like ['contrast']
    # Group by conditions and check trial counts
    group_sizes = trials_meta.groupby(conditions).size()
    valid_conditions = group_sizes[group_sizes >= min_trials_per_condition].index
    
    if len(valid_conditions) == 0:
        return np.zeros((n_pseudo, 0, n_steps, len(cluster_ids))), np.zeros((n_pseudo, 0, len(conditions)))

    # Filter trials_meta and spike_count to keep only trials in valid conditions
    trial_mask = trials_meta.set_index(conditions).index.isin(valid_conditions)
    valid_trials_meta = trials_meta[trial_mask]
    valid_spike_count = spike_count[trial_mask]

    # Bootstrapping (df_2_repsonse_matrix replacement)
    n_valid_conditions = len(valid_conditions)
    n_neurons = len(cluster_ids)
    
    response_matrix = np.zeros((n_pseudo, n_valid_conditions, n_steps, n_neurons), dtype=np.float32)
    condition_matrix = np.zeros((n_pseudo, n_valid_conditions, len(conditions)), dtype=np.float32)

    # To speed up bootstrapping, pre-group trial indices by condition
    condition_to_trial_indices = {}
    valid_trials_meta_reset = valid_trials_meta.reset_index(drop=True)
    for i, cond in enumerate(valid_conditions):
        # Handle single vs multiple conditions for indexing
        if isinstance(cond, (list, tuple, np.ndarray)) and not isinstance(cond, str):
            mask = (valid_trials_meta_reset[conditions] == cond).all(axis=1)
        else:
            # If conditions is a list with one element, or cond is a scalar
            if isinstance(conditions, list) and len(conditions) == 1:
                mask = (valid_trials_meta_reset[conditions[0]] == cond)
            else:
                mask = (valid_trials_meta_reset[conditions] == cond)
        indices = valid_trials_meta_reset.index[mask].values
        condition_to_trial_indices[cond] = indices
        condition_matrix[:, i, :] = cond

        # Pre-generate all random indices for this condition
        # (n_pseudo, len(indices))
        all_sampled_indices = np.random.choice(indices, size=(n_pseudo, len(indices)), replace=True)
        
        # Loop for each pseudo trial
        for p in range(n_pseudo):
            response_matrix[p, i, :, :] = np.mean(valid_spike_count[all_sampled_indices[p]], axis=0)
    
    del valid_spike_count
    del spike_count

    if return_cluster_ids:
        return response_matrix, condition_matrix, cluster_ids
    return response_matrix, condition_matrix


def get_region_response_matrix(region, eids, directory_path, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=5, min_trials_per_condition=10, filter_significant=False, significance_threshold=0.6, mapping='Beryl', n_pseudo=500):
    """
    Loops through eids and concatenates response_matrices for a specific region.
    Sequential implementation to avoid parallel processing overhead/crashes.
    """
    print(f"Aggregating {region} responses across {len(eids)} EIDs...")
    
    results = []
    for eid in tqdm(eids, desc="Processing EIDs"):
        res = get_eid_region_response_average(
            eid, region, directory_path, event, conditions,
            stride=stride, step=step, n_steps=n_steps,
            sigma=sigma, threshold=threshold,
            min_trials_per_condition=min_trials_per_condition,
            filter_significant=filter_significant,
            significance_threshold=significance_threshold,
            mapping=mapping,
            n_pseudo=n_pseudo
        )
        results.append(res)
    
    all_results = [r for r in results if r is not None]
    
    if not all_results:
        return None, None, None
        
    # Ensure all EIDs have the same conditions
    all_cond_matrices = [r[1] for r in all_results]

    # Helper to convert conditions to a set of tuples
    def get_cond_set(cm):
        return set(tuple(c) for c in cm[0])
    
    # Find the maximal number of conditions across all results
    cond_sets = [get_cond_set(cm) for cm in all_cond_matrices]
    max_n_conds = max(len(s) for s in cond_sets)
    
    # Filter results to only include EIDs with the maximal number of conditions
    filtered_all_results = [
        r for r, s in zip(all_results, cond_sets) if len(s) == max_n_conds
    ]
    
    # Update all_cond_matrices and cond_sets for the filtered results
    all_cond_matrices = [r[1] for r in filtered_all_results]
    cond_sets = [get_cond_set(cm) for cm in all_cond_matrices]

    # Find common conditions among the EIDs with the maximal number of conditions
    common_conditions = cond_sets[0]
    for s in cond_sets[1:]:
        common_conditions &= s
    
    if not common_conditions:
        print("No common conditions found across EIDs with maximal conditions.")
        return None, None, None
    
    # Sort common conditions to ensure consistent order
    sorted_common_conditions = sorted(list(common_conditions))
    print(f"Max conditions: {max_n_conds}, Common conditions found: {sorted_common_conditions}")

    # Build final response matrices and track EIDs
    # final_response_matrices = []
    final_neuron_eids = []
    
    # Get EIDs corresponding to filtered_all_results
    eids_filtered = [e for e, r in zip(eids, results) if r is not None and len(get_cond_set(r[1])) == max_n_conds]
    
    total_neurons = sum(r[0].shape[3] for r in filtered_all_results)
    n_conds = len(sorted_common_conditions)
    # Pre-allocate concatenated matrix to save memory during concatenation
    concatenated_matrix = np.zeros((n_pseudo, n_conds, n_steps, total_neurons), dtype=np.float32)
    
    current_neuron_idx = 0
    for (res_matrix, cond_matrix), eid in zip(filtered_all_results, eids_filtered):
        current_conds = [tuple(c) for c in cond_matrix[0]]
        indices = [current_conds.index(cc) for cc in sorted_common_conditions]
        filtered_res = res_matrix[:, indices, :, :]
        n_neurons_eid = filtered_res.shape[3]
        concatenated_matrix[:, :, :, current_neuron_idx:current_neuron_idx + n_neurons_eid] = filtered_res
        final_neuron_eids.extend([eid] * n_neurons_eid)
        current_neuron_idx += n_neurons_eid

    condition_matrix = np.zeros((n_pseudo, len(sorted_common_conditions), len(conditions)), dtype=np.float32)
    for i, cc in enumerate(sorted_common_conditions):
        condition_matrix[:, i, :] = cc
    
    print(f"Total neurons collected: {total_neurons}")
        
    return concatenated_matrix, condition_matrix, np.array(final_neuron_eids)


def get_region_eid_response_matrix(region, eids, index, directory_path, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=5, min_trials_per_condition=10, filter_significant=False, significance_threshold=0.6, mapping='Beryl', n_pseudo=500):
    """
    Returns the response matrix for a specific EID containing the region, selected by index after sorting.
    Similar to get_region_response_matrix but for a single EID.
    """
    # 1. Filter EIDs that contain the region
    valid_eids = []
    print(f"Filtering {len(eids)} EIDs for region {region}...")
    for eid in tqdm(eids):
        full_path = os.path.join(directory_path, eid)
        if not os.path.exists(full_path):
            continue
        try:
            clusters = pd.read_pickle(os.path.join(full_path, 'clusters.pkl'))
            cluster_regions = br.id2acronym(clusters['atlas_id'], mapping=mapping)
            if region in cluster_regions:
                valid_eids.append(eid)
        except Exception:
            continue

    if not valid_eids:
        print(f"No EIDs found containing region {region}")
        return None, None, None

    # 2. Sort EIDs
    valid_eids.sort()

    if index < 0 or index >= len(valid_eids):
        print(f"Index {index} out of range for {len(valid_eids)} valid EIDs.")
        return None, None, None

    selected_eid = valid_eids[index]
    print(f"Selected EID: {selected_eid} (index {index})")

    # 3. Get responses for the selected EID
    result = get_eid_region_response_average(
        selected_eid, region, directory_path, event, conditions,
        stride=stride, step=step, n_steps=n_steps,
        sigma=sigma, threshold=threshold,
        min_trials_per_condition=min_trials_per_condition,
        filter_significant=filter_significant,
        significance_threshold=significance_threshold,
        mapping=mapping,
        n_pseudo=n_pseudo
    )

    if result is None:
        return None, None, None

    res_matrix, cond_matrix = result
    neuron_eids = np.array([selected_eid] * res_matrix.shape[3])

    return res_matrix, cond_matrix, neuron_eids


def get_eid_response_matrix(eid, directory_path, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=5, min_trials_per_condition=10, filter_significant=False, significance_threshold=0.6, n_pseudo=500):
    """
    Returns the response matrix for all neurons in a specific EID.
    Also returns a list of Beryl and Cosmos region acronyms for each neuron.
    """
    full_path = os.path.join(directory_path, eid)
    if not os.path.exists(full_path):
        return None

    try:
        clusters = pd.read_pickle(os.path.join(full_path, 'clusters.pkl'))
        # Set cluster_id as index for easier lookup
        if 'cluster_id' in clusters.columns:
            clusters = clusters.drop_duplicates(subset='cluster_id')
            clusters = clusters.set_index('cluster_id')
        
        # Add Beryl and Cosmos region acronyms
        clusters['beryl'] = br.id2acronym(clusters['atlas_id'], mapping='Beryl')
        clusters['cosmos'] = br.id2acronym(clusters['atlas_id'], mapping='Cosmos')
        
        spikes = pd.read_pickle(os.path.join(full_path, 'spikes.pkl'))
        trials = pd.read_pickle(os.path.join(full_path, 'trials.pkl'))

        if filter_significant:
            event_times = trials[event].values
            event_times = event_times[~np.isnan(event_times)]
            significant_cluster_ids = filter_significant_neurons(
                spikes['times'], spikes['clusters'], event_times, threshold=significance_threshold
            )
            clusters = clusters[clusters.index.isin(significant_cluster_ids)]
            if clusters.empty:
                return None

        # Get aligned response matrix for all neurons in clusters
        # regions=None means all regions are included
        res_matrix, cond_matrix, used_cluster_ids = get_aligned_response_matrix(
            spikes, clusters, trials, event, conditions,
            stride=stride, step=step, n_steps=n_steps,
            sigma=sigma, threshold=threshold,
            min_trials_per_condition=min_trials_per_condition,
            regions=None,
            mapping_col='beryl', # mapping_col doesn't matter much when regions=None, but needs to be valid
            n_pseudo=n_pseudo,
            return_cluster_ids=True
        )

        # Filter by mean firing rate
        # res_matrix shape: (n_pseudo, n_conditions, n_steps, n_neurons)
        # The neurons in res_matrix correspond to used_cluster_ids

        if len(used_cluster_ids) == 0:
            return None
            
        mean_firing_rates = np.mean(res_matrix, axis=(0, 1, 2))
        neuron_mask = mean_firing_rates >= threshold
        res_matrix = res_matrix[:, :, :, neuron_mask]
        
        final_cluster_ids = used_cluster_ids[neuron_mask]
        
        if len(final_cluster_ids) == 0:
            return None
            
        # Get region labels for final neurons
        beryl_regions = clusters.loc[final_cluster_ids, 'beryl'].values
        cosmos_regions = clusters.loc[final_cluster_ids, 'cosmos'].values

        return res_matrix, cond_matrix, beryl_regions, cosmos_regions

    except Exception as e:
        print(f"Error processing eid {eid}: {e}")
        return None


def get_eid_region_response_average(eid, region, directory_path, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=5, min_trials_per_condition=10, filter_significant=False, significance_threshold=0.6, mapping='Beryl', n_pseudo=500):
    """
    Returns 4D array of shape n_pseudo x conditions x time x neurons representing the responses for a specific eid and region.
    """
    full_path = os.path.join(directory_path, eid)
    if not os.path.exists(full_path):
        return None

    try:
        clusters = pd.read_pickle(os.path.join(full_path, 'clusters.pkl'))
        # Set cluster_id as index for easier lookup in restrict_to_region
        if 'cluster_id' in clusters.columns:
            # Drop duplicates to ensure unique index
            clusters = clusters.drop_duplicates(subset='cluster_id')
            clusters = clusters.set_index('cluster_id')
        
        clusters['region_acronym'] = br.id2acronym(clusters['atlas_id'], mapping=mapping)
        spikes = pd.read_pickle(os.path.join(full_path, 'spikes.pkl'))
        trials = pd.read_pickle(os.path.join(full_path, 'trials.pkl'))

        # Check if region exists in this EID
        if region not in clusters['region_acronym'].unique():
            return None

        if filter_significant:
            event_times = trials[event].values
            # Filter out trials with NaN event times
            event_times = event_times[~np.isnan(event_times)]
            significant_cluster_ids = filter_significant_neurons(
                spikes['times'], spikes['clusters'], event_times, threshold=significance_threshold
            )
            # Filter clusters to keep only significant ones in the region
            # Since clusters index is now cluster_id
            clusters = clusters[clusters.index.isin(significant_cluster_ids)]
            if clusters.empty:
                return None

        res_matrix, cond_matrix = get_aligned_response_matrix(
            spikes, clusters, trials, event, conditions,
            stride=stride, step=step, n_steps=n_steps,
            sigma=sigma, threshold=threshold,
            min_trials_per_condition=min_trials_per_condition,
            regions=[region],
            mapping_col='region_acronym',
            n_pseudo=n_pseudo
        )

        # res_matrix shape: (n_pseudo, n_conditions, n_steps, n_neurons)
        
        # Filter by mean firing rate
        # We compute mean across n_pseudo, conditions, and time for each neuron
        mean_firing_rates = np.mean(res_matrix, axis=(0, 1, 2))
        res_matrix = res_matrix[:, :, :, mean_firing_rates >= threshold]

        if res_matrix.shape[3] == 0:
            return None

        return res_matrix, cond_matrix

    except Exception as e:
        print(f"Error processing eid {eid}: {e}")
        return None



# Recurrent neural network model for fitting neural responses





import matplotlib.pyplot as plt
import matplotlib

import torch

plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .5

import torch.nn.functional as F






def plot_psth(neuron_idx, data, model_data=None, condition_labels=None, title=None, ax=None):
    """
    Plots the Peri-Stimulus Time Histogram (PSTH) for a given neuron.

    Parameters
    ----------
    neuron_idx : int
        Index of the neuron to plot.
    data : array-like
        Neural data with shape (conditions, n_time, n_neurons) or (n_pseudo, conditions, n_time, n_neurons).
    model_data : array-like, optional
        Model-predicted neural data with same shape as data.
    condition_labels : list of str, optional
        Labels for each condition.
    title : str, optional
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    # If 4D (n_pseudo, n_conditions, n_time, n_neurons), we can compute error bars
    if len(data.shape) == 4:
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
    else:
        data_mean = data
        data_std = None

    if model_data is not None:
        if len(model_data.shape) == 4:
            model_mean = np.mean(model_data, axis=0)
            model_std = np.std(model_data, axis=0)
        else:
            model_mean = model_data
            model_std = None
    
    n_conditions = data_mean.shape[0]
    colormap = plt.cm.coolwarm
    colors = [colormap(i) for i in np.linspace(0, 1, n_conditions)]
    
    for i in range(n_conditions):
        label_base = condition_labels[i] if condition_labels is not None else f'Cond {i}'
        
        # Plot experimental data
        y_val = data_mean[i, :, neuron_idx]
        if hasattr(y_val, 'cpu'): y_val = y_val.cpu().detach().numpy()
        ax.plot(y_val, label=f'Data {label_base}', color=colors[i], alpha=0.8)
        
        if data_std is not None:
            y_std = data_std[i, :, neuron_idx]
            if hasattr(y_std, 'cpu'): y_std = y_std.cpu().detach().numpy()
            ax.fill_between(range(len(y_val)), y_val - y_std, y_val + y_std, color=colors[i], alpha=0.2)
        
        # Plot model prediction if provided
        if model_data is not None:
            y_model = model_mean[i, :, neuron_idx]
            if hasattr(y_model, 'cpu'): y_model = y_model.cpu().detach().numpy()
            ax.plot(y_model, label=f'Model {label_base}', color=colors[i], linestyle='--')
            
            if model_std is not None:
                y_model_std = model_std[i, :, neuron_idx]
                if hasattr(y_model_std, 'cpu'): y_model_std = y_model_std.cpu().detach().numpy()
                ax.fill_between(range(len(y_model)), y_model - y_model_std, y_model + y_model_std, 
                                color=colors[i], alpha=0.1, linestyle='--')
            
    if title is None:
        title = f'Neuron {neuron_idx}'
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Activity')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine(ax=ax)
    plt.tight_layout()
    
    return ax


