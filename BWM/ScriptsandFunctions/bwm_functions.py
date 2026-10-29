
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from scipy.ndimage import gaussian_filter1d
from iblatlas.regions import BrainRegions
br = BrainRegions()
from scipy.stats import gaussian_kde, pearsonr, wilcoxon, ttest_rel, rankdata
from scipy.interpolate import Rbf
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


def get_aligned_response_matrix(spikes, clusters, trials, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=None, std_above=None, std_below=None, min_trials_per_condition=10, regions=None, mapping_col='cosmos', n_pseudo=500, return_cluster_ids=False, start_time=-0.05, required_conditions=None):
    event_times = trials[event].values
    n_trials = len(event_times)

    # Vectorized interval generation
    starts = event_times[:, np.newaxis] + start_time + np.arange(n_steps) * step
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
    if all(c in trials_meta.columns for c in conditions):
        group_sizes = trials_meta.groupby(conditions).size()
        
        if required_conditions is not None:
            # Ensure required_conditions is a list of tuples
            required_conditions = [c if isinstance(c, (list, tuple, np.ndarray)) else (c,) for c in required_conditions]
            
            # Helper to safely match conditions even with NaNs
            def get_group_size(cond_tuple):
                mask = np.ones(len(trials_meta), dtype=bool)
                for col_idx, val in enumerate(cond_tuple):
                    col = conditions[col_idx]
                    if isinstance(val, float) and np.isnan(val):
                        mask &= trials_meta[col].isna()
                    else:
                        mask &= (trials_meta[col] == val)
                return mask.sum()

            # Check if all required conditions satisfy min_trials_per_condition
            for cond in required_conditions:
                if get_group_size(cond) < min_trials_per_condition:
                    # This EID does not satisfy one of the required conditions
                    if len(cluster_ids) == 0:
                         print(f"DEBUG: No neurons found for this EID and region.")
                    else:
                         print(f"DEBUG: EID excluded. Required condition {cond} has only {get_group_size(cond)} trials (threshold {min_trials_per_condition}).")
                    return np.zeros((n_pseudo, 0, n_steps, len(cluster_ids))), np.zeros((n_pseudo, 0, len(conditions)))
            
            valid_conditions = required_conditions
        else:
            # Fallback to behavior where we take all conditions that meet the threshold
            available_conditions = [tuple(v) if isinstance(v, (list, tuple, np.ndarray)) else (v,) for v in group_sizes[group_sizes >= min_trials_per_condition].index]
            valid_conditions = available_conditions
    else:
        print(f"Warning: some condition columns {conditions} not found in trials. Available: {trials_meta.columns}")
        valid_conditions = []
    
    if len(valid_conditions) == 0:
        if len(cluster_ids) == 0:
             print(f"DEBUG: No neurons found for this EID and region.")
        else:
             print(f"DEBUG: No valid conditions found for this EID. group_sizes for {conditions}: {trials_meta.groupby(conditions).size() if all(c in trials_meta.columns for c in conditions) else 'missing columns'}")
        return np.zeros((n_pseudo, 0, n_steps, len(cluster_ids))), np.zeros((n_pseudo, 0, len(conditions)))

    # Bootstrapping (df_2_repsonse_matrix replacement)
    n_valid_conditions = len(valid_conditions)
    n_neurons = len(cluster_ids)
    
    response_matrix = np.zeros((n_pseudo, n_valid_conditions, n_steps, n_neurons), dtype=np.float32)
    condition_matrix = np.zeros((n_pseudo, n_valid_conditions, len(conditions)), dtype=np.float32)

    # To speed up bootstrapping, pre-group trial indices by condition
    valid_trials_meta_reset = trials_meta.reset_index(drop=True)
    for i, cond in enumerate(valid_conditions):
        # Determine which rows match this full condition vector
        mask = np.ones(len(valid_trials_meta_reset), dtype=bool)
        for col_idx, val in enumerate(cond):
            if isinstance(val, float) and np.isnan(val):
                mask &= valid_trials_meta_reset[conditions[col_idx]].isna()
            else:
                mask &= (valid_trials_meta_reset[conditions[col_idx]] == val)
        
        indices = valid_trials_meta_reset.index[mask].values
        condition_matrix[:, i, :] = cond

        # Pre-generate all random indices for this condition
        # (n_pseudo, len(indices))
        all_sampled_indices = np.random.choice(indices, size=(n_pseudo, len(indices)), replace=True)
        
        # Loop for each pseudo trial
        for p in range(n_pseudo):
            response_matrix[p, i, :, :] = np.mean(spike_count[all_sampled_indices[p]], axis=0)
    
    del spike_count

    if return_cluster_ids:
        return response_matrix, condition_matrix, cluster_ids
    return response_matrix, condition_matrix

def plot_raster(eid1, eid2=None, directory_path='../Data2', start_time=None, end_time=None, fr_threshold=0.0, fig=None, gs_sub=None):
    # If eid2 is provided, we will stack two rasters for panel a.
    # If eid2 is None, we use eid1 for everything as before.
    from matplotlib import gridspec
    
    eid_for_densmap = eid1
    
    def get_data(eid):
        spikes_path = os.path.join(directory_path, eid, 'spikes.pkl')
        trials_path = os.path.join(directory_path, eid, 'trials.pkl')
        clusters_path = os.path.join(directory_path, eid, 'clusters.pkl')
        
        if not os.path.exists(spikes_path) or not os.path.exists(trials_path) or not os.path.exists(clusters_path):
            print(f"Required files not found for EID: {eid}")
            return None
            
        spikes = pd.read_pickle(spikes_path)
        trials = pd.read_pickle(trials_path)
        clusters = pd.read_pickle(clusters_path)
        return spikes, trials, clusters

    data1 = get_data(eid1)
    if data1 is None: return
    spikes1, trials1, clusters1 = data1
    
    if eid2:
        data2 = get_data(eid2)
        if data2 is None: return
        spikes2, trials2, clusters2 = data2
    else:
        spikes2, trials2, clusters2 = None, None, None

    def process_spikes(spikes, trials, clusters, start_time, end_time):
        spike_times = spikes['times']
        spike_clusters = spikes['clusters']
        
        stim_on_times = trials['stimOn_times'].values
        stim_on_times = stim_on_times[~np.isnan(stim_on_times)]

        if start_time is not None or end_time is not None:
            mask = np.ones_like(spike_times, dtype=bool)
            if start_time is not None: mask &= (spike_times >= start_time)
            if end_time is not None: mask &= (spike_times <= end_time)
            spike_times = spike_times[mask]
            spike_clusters = spike_clusters[mask]
            if start_time is not None: stim_on_times = stim_on_times[stim_on_times >= start_time]
            if end_time is not None: stim_on_times = stim_on_times[stim_on_times <= end_time]

        # Region mapping and filtering
        clusters = clusters.copy()
        clusters['beryl'] = br.id2acronym(clusters['atlas_id'], mapping='Beryl')
        clusters = clusters[~clusters['beryl'].isin(['root', 'void'])]

        # Ensure neurons have at least one spike in the window, even if fr_threshold is 0
        t_min = start_time if start_time is not None else spikes['times'].min()
        t_max = end_time if end_time is not None else spikes['times'].max()
        duration = t_max - t_min
        if duration > 0:
            counts = pd.Series(spike_clusters).value_counts()
            fr = counts / duration
            # If fr_threshold is 0, we still want to filter out neurons with 0 spikes in window
            effective_threshold = max(fr_threshold, 1e-9)
            valid_indices = fr[fr >= effective_threshold].index
            # Use .loc to select by index (which spike_clusters refers to)
            clusters = clusters.loc[clusters.index.isin(valid_indices)]
        elif fr_threshold > 0:
            clusters = clusters.iloc[0:0]

        clusters_sorted = clusters.sort_values(['beryl', 'depths'], ascending=[True, False])
        clusters_sorted['plot_coord'] = np.arange(len(clusters_sorted))
        # id_to_coord mapping must use the index of clusters, because spike_clusters contains indices
        id_to_coord = {idx: coord for idx, coord in zip(clusters_sorted.index, clusters_sorted.plot_coord)}
        spike_coords = np.array([id_to_coord.get(c, -1) for c in spike_clusters])
        valid_mask = spike_coords != -1
        # reset_index for clusters_sorted for easier iteration later if needed
        clusters_sorted = clusters_sorted.reset_index(drop=True)
        return spike_times[valid_mask], spike_coords[valid_mask], stim_on_times, clusters, clusters_sorted, id_to_coord

    st1, sc1, son1, c1, cs1, i2c1 = process_spikes(spikes1, trials1, clusters1, start_time, end_time)
    if eid2:
        st2, sc2, son2, c2, cs2, i2c2 = process_spikes(spikes2, trials2, clusters2, start_time, end_time)
    else:
        st2, sc2, son2, c2, cs2, i2c2 = None, None, None, None, None, None

    if fig is None:
        fig = plt.figure(figsize=(10, 6))
        gs_parent = gridspec.GridSpec(1, 1, figure=fig)
        gs_sub = gs_parent[0]
    
    gs_main = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_sub, height_ratios=[2, 1], hspace=0.4)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0], width_ratios=[1.2, 1, 1, 1], wspace=0.3)
    
    # Raster Group: Split into two vertically if eid2 exists
    if eid2:
        gs_raster_area = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_top[0], hspace=0.1)
        gs_raster_group1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_raster_area[0], width_ratios=[0.05, 1], wspace=0.05)
        gs_raster_group2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_raster_area[1], width_ratios=[0.05, 1], wspace=0.05)
        ax_bar1 = fig.add_subplot(gs_raster_group1[0])
        ax_raster1 = fig.add_subplot(gs_raster_group1[1])
        ax_bar2 = fig.add_subplot(gs_raster_group2[0])
        ax_raster2 = fig.add_subplot(gs_raster_group2[1])
        raster_axes = [(ax_bar1, ax_raster1, cs1, st1, sc1, son1, eid1), (ax_bar2, ax_raster2, cs2, st2, sc2, son2, eid2)]
    else:
        gs_raster_group = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_top[0], width_ratios=[0.05, 1], wspace=0.05)
        ax_bar1 = fig.add_subplot(gs_raster_group[0])
        ax_raster1 = fig.add_subplot(gs_raster_group[1])
        raster_axes = [(ax_bar1, ax_raster1, cs1, st1, sc1, son1, eid1)]

    ax_pca = fig.add_subplot(gs_top[1])
    ax_densmap = fig.add_subplot(gs_top[2])
    ax_density = fig.add_subplot(gs_top[3])
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_main[1], wspace=0.3)
    axes_neurons = [fig.add_subplot(gs_bottom[i]) for i in range(5)]
    ax_neuron1 = axes_neurons[0]
    
    # Panel C & DensMAP use both EIDs if available
    # For DensMAP, we use the original spikes (not window-filtered) to ensure it uses all time points
    
    # Use union of regions for color mapping consistency if needed, but per-plot is fine
    all_clusters = pd.concat([c1, c2]) if eid2 and c2 is not None else c1

    unique_regions = sorted(all_clusters['beryl'].unique())
    # Use Set2 colormap as requested
    cmap = plt.get_cmap('Set2')
    palette = [cmap(i % cmap.N) for i in range(len(unique_regions))]
    region_to_color = {region: palette[i] for i, region in enumerate(unique_regions)}

    for ax_bar, ax_raster, cs, st, sc, son, eid in raster_axes:
        # Sort spikes by cluster coord to avoid overplotting artifacts if needed, 
        # but discrete spikes are requested. Use small markers.
        ax_raster.scatter(st, sc, s=0.2, c='black', marker='o', alpha=0.3, edgecolors='none', linewidths=0)
        for _, row in cs.iterrows():
            ax_bar.barh(row['plot_coord'], 1, color=region_to_color[row['beryl']], height=1.0, edgecolor='none')
        
        # Region labels
        u_regs = sorted(cs['beryl'].unique())
        for region in u_regs:
            region_indices = cs[cs['beryl'] == region].index
            if len(region_indices) > 0:
                diffs = np.diff(region_indices)
                breaks = np.where(diffs > 1)[0]
                start_idx = 0
                for b in list(breaks) + [len(region_indices) - 1]:
                    block = region_indices[start_idx:b+1]
                    mid_pos = np.mean(block)
                    ax_bar.text(-0.5, mid_pos, region, va='center', ha='right', fontsize=6)
                    start_idx = b + 1
        
        ax_bar.set_ylim(-0.5, len(cs) - 0.5)
        ax_bar.set_xlim(-1, 1)
        ax_bar.axis('off')
        for t in son:
            ax_raster.axvspan(t, t + 0.5, color='gray', alpha=0.3, lw=0, zorder=0)
        if start_time is not None or end_time is not None:
            ax_raster.set_xlim(left=start_time, right=end_time)
        ax_raster.set_ylim(-0.5, len(cs) - 0.5)
        ax_raster.set_xlabel('Time (s)', fontsize=6)
        ax_raster.tick_params(axis='x', labelsize=6)
        ax_raster.set_yticks([])
        ax_raster.set_yticklabels([])
        ax_raster.set_yticklabels([])

    res_matrix1, cond_matrix1, cluster_ids1 = get_aligned_response_matrix(
        spikes1, c1, trials1, event='stimOn_times', conditions=['choice'], 
        n_steps=100, step=0.005, stride=0.05, n_pseudo=25, min_trials_per_condition=10,
        mapping_col='beryl', return_cluster_ids=True
    )
    
    if eid2:
        res_matrix2, cond_matrix2, cluster_ids2 = get_aligned_response_matrix(
            spikes2, c2, trials2, event='stimOn_times', conditions=['choice'], 
            n_steps=100, step=0.005, stride=0.05, n_pseudo=25, min_trials_per_condition=10,
            mapping_col='beryl', return_cluster_ids=True
        )
        # Combine matrices if they have compatible shapes
        if res_matrix1.shape[1] == res_matrix2.shape[1] and np.array_equal(cond_matrix1[0], cond_matrix2[0]):
            res_matrix = np.concatenate([res_matrix1, res_matrix2], axis=3)
            cluster_ids = np.concatenate([cluster_ids1, cluster_ids2])
            cond_matrix = cond_matrix1
            # Keep track of which neuron belongs to which EID for mapping back to coordinates
            is_from_eid2 = np.array([False] * len(cluster_ids1) + [True] * len(cluster_ids2))
        else:
            print("Warning: EID1 and EID2 have incompatible conditions for merging. Using EID1 only.")
            res_matrix, cond_matrix, cluster_ids = res_matrix1, cond_matrix1, cluster_ids1
            is_from_eid2 = np.array([False] * len(cluster_ids1))
    else:
        res_matrix, cond_matrix, cluster_ids = res_matrix1, cond_matrix1, cluster_ids1
        is_from_eid2 = np.array([False] * len(cluster_ids1))
    
    # Select example neurons (from combined pool)
    if res_matrix.shape[3] >= 5:
        # Note: We already filtered by fr_threshold in the window within process_spikes.
        # We can still filter here for 'active' neurons in terms of response matrix,
        # but the request was to align with the window filtering.
        # For now, we'll keep this but ensure it doesn't conflict too much.
        mean_fr = np.mean(res_matrix, axis=(0, 1, 2))
        active_neurons = np.where(mean_fr >= 0)[0] # Keep all that survived process_spikes
        if len(active_neurons) < 5: active_neurons = np.arange(res_matrix.shape[3])
        active_cluster_ids = cluster_ids[active_neurons]
        active_regions = []
        filtered_active_neurons = []
        for i, cid in enumerate(active_cluster_ids):
            if is_from_eid2[active_neurons[i]]:
                source_c = c2
            else:
                source_c = c1
            matches = source_c[source_c['cluster_id'] == cid]['beryl']
            if not matches.empty:
                region = matches.iloc[0]
                if region != 'void':
                    active_regions.append(region)
                    filtered_active_neurons.append(active_neurons[i])
        active_neurons = filtered_active_neurons
        
        found_different = False
        if len(set(active_regions)) >= 5:
            from collections import defaultdict
            region_to_indices = defaultdict(list)
            for i, region in enumerate(active_regions):
                region_to_indices[region].append(active_neurons[i])
            available_regions = list(region_to_indices.keys())
            selected_regions = np.random.choice(available_regions, 5, replace=False)
            selected_indices = np.array([np.random.choice(region_to_indices[r]) for r in selected_regions])
            found_different = True
        if not found_different:
            selected_indices = np.random.choice(active_neurons, min(5, len(active_neurons)), replace=False)
        
        example_coords = []
        example_colors = []
        for idx in selected_indices:
            cid = cluster_ids[idx]
            if is_from_eid2[idx]:
                source_i2c = i2c2
                source_c = c2
            else:
                source_i2c = i2c1
                source_c = c1
            
            # Since cid here is the index (from cluster_ids which came from get_aligned_response_matrix),
            # we can look it up directly in source_i2c.
            coord = source_i2c.get(cid)
            example_coords.append(coord)
            if coord is not None:
                # source_c might be the original clusters or filtered one, but we use .loc with cid index
                if cid in source_c.index:
                    region = source_c.loc[cid, 'beryl']
                    example_colors.append(region_to_color.get(region, 'black'))
                else:
                    example_colors.append('black')
            else:
                example_colors.append('black')
        
        for i, (coord, color) in enumerate(zip(example_coords, example_colors)):
            if coord is not None:
                idx = selected_indices[i]
                if is_from_eid2[idx]:
                    target_ax = ax_raster2
                    target_st = st2
                    target_sc = sc2
                else:
                    target_ax = ax_raster1
                    target_st = st1
                    target_sc = sc1
                mask = target_sc == coord
                target_ax.scatter(target_st[mask], target_sc[mask], s=0.3, color=color, marker='o', alpha=.3, edgecolors='none', linewidths=0)

        # Update right-side labels for both rasters if they have selected neurons
        raster_axes_for_labels = [(ax_raster1, False)]
        if eid2:
            raster_axes_for_labels.append((ax_raster2, True))
            
        for ax_r, is_eid2 in raster_axes_for_labels:
            ax_r_right = ax_r.twinx()
            ax_r_right.set_ylim(ax_r.get_ylim())
            relevant_indices = [i for i, idx in enumerate(selected_indices) if is_from_eid2[idx] == is_eid2 and example_coords[i] is not None]
            if relevant_indices:
                ax_r_right.set_yticks([example_coords[i] for i in relevant_indices])
                ax_r_right.set_yticklabels([f"{selected_indices[i]}" for i in relevant_indices], fontsize=5)
            else:
                ax_r_right.set_yticks([])
            ax_r_right.axis('off')

        all_z_values = []
        neuron_dfs = []
        for neuron_idx in selected_indices:
            z_neuron = res_matrix[:, :, :, neuron_idx]
            z_neuron = (z_neuron - np.mean(z_neuron)) / (np.std(z_neuron) + 1e-8)
            all_z_values.append(z_neuron)
            rows = []
            for p in range(z_neuron.shape[0]):
                for c in range(z_neuron.shape[1]):
                    for t in range(z_neuron.shape[2]):
                        rows.append({'pseudo_trial': p, 'condition': cond_matrix[0, c, 0], 't': t * 0.005, 'z': z_neuron[p, c, t]})
            neuron_dfs.append(pd.DataFrame(rows))
            
        global_ymin = min([z.min() for z in all_z_values])
        global_ymax = max([z.max() for z in all_z_values])
        y_range = global_ymax - global_ymin
        global_ymin -= 0.05 * y_range
        global_ymax += 0.05 * y_range

        for i, (ax, neuron_idx, df_neuron) in enumerate(zip(axes_neurons, selected_indices, neuron_dfs)):
            cid = cluster_ids[neuron_idx]
            if is_from_eid2[neuron_idx]:
                source_c = c2
            else:
                source_c = c1
            
            if cid in source_c.index:
                region = source_c.loc[cid, 'beryl']
                title_color = region_to_color.get(region, "black")
            else:
                title_color = "black"
            # Plot using solid and dashed lines for conditions instead of color
            for condition_idx, condition_val in enumerate(df_neuron['condition'].unique()):
                cond_df = df_neuron[df_neuron['condition'] == condition_val]
                linestyle = '-' if condition_idx == 0 else '--'
                
                # Plot individual pseudo-trials
                sns.lineplot(data=cond_df, ax=ax, x='t', y='z', units='pseudo_trial', 
                             estimator=None, legend=False, lw=0.25, alpha=0.3, 
                             color='black', linestyle=linestyle)
                
                # Plot mean response
                sns.lineplot(data=cond_df, ax=ax, x='t', y='z', 
                             legend=False, lw=1, alpha=1.0, 
                             color='black', linestyle=linestyle)
            
            ax.set_title(f"{neuron_idx}", fontsize=6, color=title_color, pad=2)
            ax.spines['bottom'].set_position(('data', 0))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Time (s)', fontsize=6)
            ax.set_ylim(global_ymin, global_ymax)
            if i == 0:
                ax.set_ylabel('z-score', fontsize=6)
                ax.tick_params(axis='both', which='major', labelsize=5)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
                ax.tick_params(axis='x', which='major', labelsize=5)
                ax.tick_params(axis='y', which='major', labelsize=0, left=True)
        
        
        # Adjust panel label positions for the new layout
        ax_raster1.text(-0.1, 1.05, 'a', transform=ax_raster1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        ax_pca.text(-0.1, 1.05, 'b', transform=ax_pca.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        ax_densmap.text(-0.1, 1.05, 'c', transform=ax_densmap.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        ax_density.text(-0.1, 1.05, 'd', transform=ax_density.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        ax_neuron1.text(-0.1, 1.05, 'e', transform=ax_neuron1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        
        # --- DensMAP Projection (all neurons) ---
        # 1. Identify all regions for color mapping
        neuron_to_region = []
        for i, cid in enumerate(cluster_ids):
            if is_from_eid2[i]:
                source_c = c2
            else:
                source_c = c1
            matches = source_c[source_c['cluster_id'] == cid]['beryl']
            if not matches.empty:
                region = matches.iloc[0]
                neuron_to_region.append(region)
            else:
                neuron_to_region.append('unknown')
        
        target_neuron_indices = [i for i, region in enumerate(neuron_to_region) if region not in ['root', 'void', 'unknown']]
        
        if len(target_neuron_indices) > 2:
            import umap
            from sklearn.decomposition import PCA
            res_matrix_target = res_matrix[:, :, :, target_neuron_indices]
            mean_res = np.mean(res_matrix_target, axis=0) 
            mean_res_norm = (mean_res - np.mean(mean_res, axis=(0, 1), keepdims=True)) / (np.std(mean_res, axis=(0, 1), keepdims=True) + 1e-8)
            X_pop = mean_res_norm.reshape(-1, mean_res_norm.shape[2])
            X_neurons = X_pop.T
            X_densmap_input = X_neurons / (np.linalg.norm(X_neurons, axis=1, keepdims=True) + 1e-8)
            
            # --- PCA Projection ---
            pca = PCA(n_components=2)
            pca_embedding = pca.fit_transform(X_densmap_input)
            ax_pca.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c='gray', s=1, alpha=0.5)
            
            # Highlight selected neurons on PCA
            for idx_in_all, color in zip(selected_indices, example_colors):
                if idx_in_all in target_neuron_indices:
                    idx_in_target = target_neuron_indices.index(idx_in_all)
                    ax_pca.scatter(pca_embedding[idx_in_target, 0], pca_embedding[idx_in_target, 1], c=[color], s=10, alpha=1.0, edgecolors='white', linewidths=0.5)
            
            # --- DensMAP Projection ---
            reducer = umap.UMAP(densmap=True, dens_lambda=10, n_neighbors=min(50, X_densmap_input.shape[0]-1))
            embedding = reducer.fit_transform(X_densmap_input)
            
            # Use gray for all neurons in DensMAP as requested
            ax_densmap.scatter(embedding[:, 0], embedding[:, 1], c='gray', s=1, alpha=0.5)
            
            # Highlight selected neurons
            for idx_in_all, color in zip(selected_indices, example_colors):
                if idx_in_all in target_neuron_indices:
                    idx_in_target = target_neuron_indices.index(idx_in_all)
                    ax_densmap.scatter(embedding[idx_in_target, 0], embedding[idx_in_target, 1], c=[color], s=10, alpha=1.0, edgecolors='white', linewidths=0.5)
                    ax_densmap.text(embedding[idx_in_target, 0] + 0.1, embedding[idx_in_target, 1] + 0.1, f"{idx_in_all}", fontsize=6, fontweight='bold', color=color)
            
            # --- Density RBF Plot ---
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(embedding.T)
            density = kde(embedding.T)
            
            # Normalize density for plotting
            density_norm = (density - density.min()) / (density.max() - density.min())
            
            # RBF Interpolation for smooth background
            x_min, x_max = embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5
            y_min, y_max = embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5
            XI, YI = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            
            try:
                rbf = Rbf(embedding[:, 0], embedding[:, 1], density_norm, function='multiquadric', smooth=1)
                ZI = rbf(XI, YI)
                # Overlay densmap projection in gray
                ax_density.scatter(embedding[:, 0], embedding[:, 1], c='gray', s=1, alpha=0.5, zorder=1)
                ax_density.contourf(XI, YI, ZI, levels=50, cmap='Spectral', alpha=0.8, zorder=2)
            except Exception as e:
                print(f"RBF failed: {e}")
                # Overlay densmap projection in gray
                ax_density.scatter(embedding[:, 0], embedding[:, 1], c='gray', s=1, alpha=0.5, zorder=1)
                ax_density.scatter(embedding[:, 0], embedding[:, 1], c=density_norm, s=1, cmap='Spectral', alpha=0.8, zorder=2)
            
            # Highlight selected neurons on density plot as well
            for idx_in_all, color in zip(selected_indices, example_colors):
                if idx_in_all in target_neuron_indices:
                    idx_in_target = target_neuron_indices.index(idx_in_all)
                    ax_density.scatter(embedding[idx_in_target, 0], embedding[idx_in_target, 1], c=[color], s=10, alpha=1.0, edgecolors='white', linewidths=0.5, zorder=3)
            
            ax_pca.set_title('PCA', fontsize=7)
            ax_densmap.set_title('DensMAP', fontsize=7)
            ax_density.set_title('Local Density', fontsize=7)
        else:
            ax_pca.set_title('PCA', fontsize=7)
            ax_pca.text(0.5, 0.5, 'Not enough neurons', ha='center', va='center')
            ax_densmap.set_title('DensMAP', fontsize=7)
            ax_densmap.text(0.5, 0.5, 'Not enough neurons', ha='center', va='center')
            ax_density.axis('off')

        ax_pca.set_xlabel('PC 1', fontsize=6)
        ax_pca.set_ylabel('PC 2', fontsize=6)
        ax_pca.set_xticks([]); ax_pca.set_yticks([])
        for spine in ax_pca.spines.values(): spine.set_visible(False)

        ax_densmap.set_xlabel('dMAP 1', fontsize=6)
        ax_densmap.set_ylabel('dMAP 2', fontsize=6)
        ax_densmap.set_xticks([]); ax_densmap.set_yticks([])
        for spine in ax_densmap.spines.values(): spine.set_visible(False)

        ax_density.set_xlabel('dMAP 1', fontsize=6)
        ax_density.set_ylabel('dMAP 2', fontsize=6)
        ax_density.set_xticks([]); ax_density.set_yticks([])
        for spine in ax_density.spines.values(): spine.set_visible(False)

        if gs_sub is None:
            os.makedirs('../Figures', exist_ok=True)
            fig_path = os.path.join('../Figures', f'BWMSchematic_{eid1}.jpg')
            fig.savefig(fig_path, dpi=600, bbox_inches='tight')
            print(f"Figure saved to {fig_path}")

    if gs_sub is None: plt.show()
    return fig


def get_region_response_matrix(region, eids, directory_path, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=None, std_above=None, std_below=None, min_trials_per_condition=10, filter_significant=False, significance_threshold=0.6, mapping='Beryl', n_pseudo=500, start_time=-0.05, required_conditions=None):
    """
    Loops through eids and concatenates response_matrices for a specific region.
    Sequential implementation to avoid parallel processing overhead/crashes.
    """
    # region can be a single string or a list of strings
    if isinstance(region, str):
        region_list = [region]
    else:
        region_list = list(region)
    
    print(f"Aggregating {region} responses across {len(eids)} EIDs...")
    
    if required_conditions is not None:
        required_conditions = [c if isinstance(c, (list, tuple, np.ndarray)) else (c,) for c in required_conditions]
    
    results = []
    for eid in tqdm(eids, desc="Processing EIDs"):
        res = get_eid_region_response_average(
            eid, region_list, directory_path, event, conditions,
            stride=stride, step=step, n_steps=n_steps,
            sigma=sigma, threshold=threshold,
            std_above=std_above, std_below=std_below,
            min_trials_per_condition=min_trials_per_condition,
            filter_significant=filter_significant,
            significance_threshold=significance_threshold,
            mapping=mapping,
            n_pseudo=n_pseudo,
            start_time=start_time,
            required_conditions=required_conditions,
            return_cluster_ids=True
        )
        results.append(res)
    
    all_results = [r for r in results if r is not None]
    
    if not all_results:
        return None, None, None
        
    # Ensure all EIDs have the same conditions
    all_cond_matrices = [r[1] for r in all_results]

    # Helper to convert conditions to a set of tuples
    def get_cond_set(cm):
        # Using np.nan in tuples might be tricky for set intersection
        # Convert nan to a special value for set operations if needed, but 
        # here we just want to ensure consistency.
        return set(tuple(c) for c in cm[0])
    
    # In stacked mode, we want to find the union of all conditions across EIDs
    # and then fill in zeros for missing conditions in each EID?
    # Or find common ones? The user's request doesn't explicitly say.
    # Given the previous implementation found common conditions, we'll stick to that
    # but be careful with NaNs.
    
    def get_cond_set_safe(cm):
        if cm is None or len(cm) == 0:
            return set()
        # Convert NaNs to a representative string or value for set intersection
        # and ensure other values are Python types to avoid UFuncTypeError during sorting
        def to_safe(x):
            if isinstance(x, (float, np.floating)) and np.isnan(x):
                return 'NaN'
            if isinstance(x, (np.generic, np.ndarray)):
                return x.item()
            return x
        return set(tuple(to_safe(x) for x in c) for c in cm[0])

    cond_sets = [get_cond_set_safe(cm) for cm in all_cond_matrices]
    
    # Find common conditions
    # For multiple conditions, we want the intersection to ensure consistency across EIDs
    # unless some flexibility is desired.
    common_conditions_safe = cond_sets[0]
    for s in cond_sets[1:]:
        common_conditions_safe &= s
    
    if not common_conditions_safe:
        # If intersection is empty, maybe fallback to union but with a warning
        print("Warning: No common conditions found across EIDs using intersection. Falling back to union.")
        common_conditions_safe = set()
        for s in cond_sets:
            common_conditions_safe |= s
    
    if not common_conditions_safe:
        print("No conditions found across EIDs.")
        return None, None, None
    
    # Convert back from 'NaN' string to np.nan
    sorted_common_conditions = []
    # Sort with a key that converts all elements to strings to avoid mixed-type comparison errors
    for c_safe in sorted(list(common_conditions_safe), key=lambda x: [str(y) for y in x]):
        sorted_common_conditions.append(tuple(np.nan if x == 'NaN' else x for x in c_safe))
    
    print(f"Final conditions count: {len(sorted_common_conditions)}")

    # Build final response matrices and track EIDs
    final_neuron_eids = []
    final_cluster_ids = []
    
    # Get EIDs corresponding to all_results
    eids_with_results = [e for e, r in zip(eids, results) if r is not None]
    
    total_neurons = sum(r[0].shape[3] for r in all_results)
    n_conds = len(sorted_common_conditions)
    # Pre-allocate concatenated matrix
    concatenated_matrix = np.zeros((n_pseudo, n_conds, n_steps, total_neurons), dtype=np.float32)
    
    current_neuron_idx = 0
    used_eids = []
    for (res_matrix, cond_matrix, used_ids), eid in zip(all_results, eids_with_results):
        current_conds = [tuple(c) for c in cond_matrix[0]]
        
        # Safe matching for conditions with NaNs
        indices = []
        found_mask = []
        for cc in sorted_common_conditions:
            # Find index of cc in current_conds, handling NaNs
            found_idx = -1
            for idx, c in enumerate(current_conds):
                if len(c) != len(cc): continue
                match = True
                for val1, val2 in zip(c, cc):
                    if np.isnan(val1) and np.isnan(val2):
                        continue
                    if val1 != val2:
                        match = False
                        break
                if match:
                    found_idx = idx
                    break
            
            indices.append(found_idx)
            found_mask.append(found_idx != -1)
        
        indices = np.array(indices)
        found_mask = np.array(found_mask)
        
        n_neurons_eid = res_matrix.shape[3]
        
        valid_indices = indices[found_mask]
        concatenated_matrix[:, found_mask, :, current_neuron_idx:current_neuron_idx + n_neurons_eid] = res_matrix[:, valid_indices, :, :]
        
        final_neuron_eids.extend([eid] * n_neurons_eid)
        final_cluster_ids.extend(used_ids)
        used_eids.append(eid)
        current_neuron_idx += n_neurons_eid

    # Trim concatenated_matrix if we skipped some neurons
    if current_neuron_idx < total_neurons:
        concatenated_matrix = concatenated_matrix[:, :, :, :current_neuron_idx]

    condition_matrix = np.zeros((n_pseudo, n_conds, len(conditions)), dtype=np.float32)
    for i, cc in enumerate(sorted_common_conditions):
        condition_matrix[:, i, :] = cc
    
    print(f"Total neurons collected: {current_neuron_idx}")
        
    return concatenated_matrix, condition_matrix, np.array(final_neuron_eids)


def get_region_eid_response_matrix(region, eids, index, directory_path, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=None, std_above=None, std_below=None, min_trials_per_condition=10, filter_significant=False, significance_threshold=0.6, mapping='Beryl', n_pseudo=500, start_time=-0.05, required_conditions=None):
    """
    Returns the response matrix for a specific EID containing the region, selected by index after sorting.
    Similar to get_region_response_matrix but for a single EID.
    """
    # 1. Filter EIDs that contain any of the regions
    if isinstance(region, str):
        region_list = [region]
    else:
        region_list = list(region)

    valid_eids = []
    print(f"Filtering {len(eids)} EIDs for region {region}...")
    for eid in tqdm(eids):
        full_path = os.path.join(directory_path, eid)
        if not os.path.exists(full_path):
            continue
        try:
            clusters = pd.read_pickle(os.path.join(full_path, 'clusters.pkl'))
            cluster_regions = br.id2acronym(clusters['atlas_id'], mapping=mapping)
            if any(r in cluster_regions for r in region_list):
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
        selected_eid, region_list, directory_path, event, conditions,
        stride=stride, step=step, n_steps=n_steps,
        sigma=sigma, threshold=threshold,
        std_above=std_above, std_below=std_below,
        min_trials_per_condition=min_trials_per_condition,
        filter_significant=filter_significant,
        significance_threshold=significance_threshold,
        mapping=mapping,
        n_pseudo=n_pseudo,
        start_time=start_time,
        required_conditions=required_conditions,
        return_cluster_ids=True
    )

    if result is None:
        return None, None, None

    res_matrix, cond_matrix, used_cluster_ids = result
    neuron_eids = np.array([selected_eid] * res_matrix.shape[3])

    return res_matrix, cond_matrix, neuron_eids


def get_eid_response_matrix(eid, directory_path, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=None, std_above=None, std_below=None, min_trials_per_condition=10, filter_significant=False, significance_threshold=0.6, n_pseudo=500, start_time=-0.05, required_conditions=None):
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
        
        # Add prev_choice condition
        trials['prev_choice'] = trials['choice'].shift(1)

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
            std_above=std_above, std_below=std_below,
            min_trials_per_condition=min_trials_per_condition,
            regions=None,
            mapping_col='beryl', # mapping_col doesn't matter much when regions=None, but needs to be valid
            n_pseudo=n_pseudo,
            return_cluster_ids=True,
            start_time=start_time,
            required_conditions=required_conditions
        )

        # Filter by mean firing rate
        # res_matrix shape: (n_pseudo, n_conditions, n_steps, n_neurons)
        # The neurons in res_matrix correspond to used_cluster_ids

        if len(used_cluster_ids) == 0:
            return None
            
        mean_firing_rates = np.mean(res_matrix, axis=(0, 1, 2))
        
        if threshold is not None:
            neuron_mask = mean_firing_rates >= threshold
        elif std_above is not None or std_below is not None:
            log_firing_rates = np.log(mean_firing_rates + 1e-6)
            avg_fr = np.mean(log_firing_rates)
            std_fr = np.std(log_firing_rates)
            neuron_mask = np.ones(len(log_firing_rates), dtype=bool)
            if std_above is not None:
                neuron_mask &= (log_firing_rates <= avg_fr + std_above * std_fr)
            if std_below is not None:
                neuron_mask &= (log_firing_rates >= avg_fr - std_below * std_fr)
        else:
            # Default to original behavior if nothing is provided? 
            # Or just don't filter. The original default was 5.
            neuron_mask = mean_firing_rates >= 5
            
        res_matrix = res_matrix[:, :, :, neuron_mask]
        
        final_cluster_ids = used_cluster_ids[neuron_mask]
        
        if len(final_cluster_ids) == 0:
            return None
            
        # Get region labels for final neurons
        beryl_regions = clusters.loc[final_cluster_ids, 'beryl'].values
        cosmos_regions = clusters.loc[final_cluster_ids, 'cosmos'].values

        return res_matrix, cond_matrix, beryl_regions, cosmos_regions, final_cluster_ids

    except Exception as e:
        print(f"Error processing eid {eid}: {e}")
        return None


def get_eid_region_response_average(eid, region, directory_path, event, conditions, stride=.01, step=.01, n_steps=100, sigma=5, threshold=None, std_above=None, std_below=None, min_trials_per_condition=10, filter_significant=False, significance_threshold=0.6, mapping='Beryl', n_pseudo=500, start_time=-0.05, required_conditions=None, return_cluster_ids=False):
    """
    Returns 4D array of shape n_pseudo x conditions x time x neurons representing the responses for a specific eid and region(s).
    region can be a string or a list of strings.
    """
    if isinstance(region, str):
        region_list = [region]
    else:
        region_list = list(region)

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
        
        # Add prev_choice condition
        trials['prev_choice'] = trials['choice'].shift(1)

        # Check if any of the regions exist in this EID
        if not any(r in clusters['region_acronym'].unique() for r in region_list):
            return None

        if filter_significant:
            event_times = trials[event].values
            # Filter out trials with NaN event times
            event_times = event_times[~np.isnan(event_times)]
            significant_cluster_ids = filter_significant_neurons(
                spikes['times'], spikes['clusters'], event_times, threshold=significance_threshold
            )
            # Filter clusters to keep only significant ones in the region(s)
            # Since clusters index is now cluster_id
            clusters = clusters[clusters.index.isin(significant_cluster_ids)]
            if clusters.empty:
                return None

        res_matrix, cond_matrix, used_cluster_ids = get_aligned_response_matrix(
            spikes, clusters, trials, event, conditions,
            stride=stride, step=step, n_steps=n_steps,
            sigma=sigma, threshold=threshold,
            std_above=std_above, std_below=std_below,
            min_trials_per_condition=min_trials_per_condition,
            regions=region_list,
            mapping_col='region_acronym',
            n_pseudo=n_pseudo,
            start_time=start_time,
            required_conditions=required_conditions,
            return_cluster_ids=True
        )

        # res_matrix shape: (n_pseudo, n_conditions, n_steps, n_neurons)
        
        # Filter by mean firing rate
        # We compute mean across n_pseudo, conditions, and time for each neuron
        mean_firing_rates = np.mean(res_matrix, axis=(0, 1, 2))
        
        if threshold is not None:
            firing_mask = mean_firing_rates >= threshold
        elif std_above is not None or std_below is not None:
            log_firing_rates = np.log(mean_firing_rates + 1e-6)
            avg_fr = np.mean(log_firing_rates)
            std_fr = np.std(log_firing_rates)
            firing_mask = np.ones(len(log_firing_rates), dtype=bool)
            if std_above is not None:
                firing_mask &= (log_firing_rates <= avg_fr + std_above * std_fr)
            if std_below is not None:
                firing_mask &= (log_firing_rates >= avg_fr - std_below * std_fr)
        else:
            # Default to original behavior
            firing_mask = mean_firing_rates >= 5
            
        res_matrix = res_matrix[:, :, :, firing_mask]
        used_cluster_ids = used_cluster_ids[firing_mask]

        if res_matrix.shape[3] == 0:
            return None

        if return_cluster_ids:
            return res_matrix, cond_matrix, used_cluster_ids
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


