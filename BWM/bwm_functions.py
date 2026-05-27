
import pandas as pd
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
from scipy.stats import gaussian_kde, pearsonr
from scipy.sparse import save_npz, load_npz, csr_matrix

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

def get_spike_counts_in_bins(spike_times, spike_clusters, intervals):
    """
    Return the number of spikes in a sequence of time intervals, for each neuron.

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    intervals : 2D array of shape (n_events, 2)
        the start and end times of the events

    Returns
    ---------
    counts : 2D array of shape (n_neurons, n_events)
        the spike counts of all neurons ffrom scipy.stats import sem, tor all events
        value (i, j) is the number of spikes of neuron `neurons[i]` in interval #j
    cluster_ids : 1D array
        list of cluster ids
    """

    # Check input
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2
    assert np.all(np.diff(spike_times) >= 0), "Spike times need to be sorted"

    intervals_idx = np.searchsorted(spike_times, intervals)

    # For each neuron and each interval, the number of spikes in the interval.
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    n_intervals = intervals.shape[0]
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        i0, i1 = intervals_idx[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(spike_clusters[i0:i1], minlength=cluster_ids.max() + 1)
        counts[:, j] = x[cluster_ids]
    return counts, cluster_ids




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


def restrict_to_region(spikes,cluster_df,regions):
    # within_region_cluster_ids = cluster_df[cluster_df.cosmos==cosmos].cluster_id.values
    # all_cluster_ids = spikes['clusters']
   # idx = np.argwhere(cluster_df['cosmos'][spikes['clusters']].isin(cosmos))[:,0]
    idx = [i for i, x in enumerate(cluster_df['cosmos'][spikes['clusters']]) if x in regions]
    spike_times = spikes['times'][idx]
    spike_clusters= spikes['clusters'][idx]
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


def get_aligned_response_matrix(spikes,clusters, trials, event,conditions, stride=.01, step=.01, n_steps=100,sigma = 5,threshold = 5,min_trials_per_condition = 10,regions=None):

    event_times = trials[event]
    n_trials = len(event_times)

    # Get spike counts in bins aligned to event times.
    intervals = []
    for trial in range(n_trials):
        start = event_times[trial] - .05
        for _ in range (n_steps):
            intervals.append([start, start + stride])
            start += step
    intervals = np.stack(intervals)
    spike_clusters,spike_times = restrict_to_region(spikes,clusters,regions)
    spike_count, cluster_id = get_spike_counts_in_bins(spike_times, spike_clusters,intervals= intervals)

    # Reshape spike_counts in 3D array
    spike_count = spike_count.T.reshape(n_trials,n_steps,-1)

   # response_matrix = np.zeros(spike_count.shape)
    # for neuron in range(spike_count.shape[2]):
    #     for trial in range(spike_count.shape[0]):
    #         response_matrix[trial,:,neuron] = gaussian_filter1d(spike_count[trial,:,neuron]/stride,sigma = sigma)

    rows = []
    contrasts = np.nan_to_num(-trials['contrastLeft'], copy=True, nan=1.0) * np.nan_to_num(
    trials['contrastRight'], copy=True, nan=1.0)
    for trial in range(1,n_trials):
        rows.append({'trial':trial,
                     'choice': trials['choice'][trial],
                     'prev_choice': trials['choice'][trial-1],
                     'reward': trials['feedbackType'][trial],
                     'prev_reward': trials['feedbackType'][trial-1],
                     'contrast': contrasts[trial],
                     'block': trials['probabilityLeft'][trial],
                     'side': np.sign(contrasts[trial]),
                     'prev_side': np.sign(contrasts[trial-1]),
                     'response': spike_count[trial,:,:],})

    # Remove inactive neurons
    df = pd.DataFrame(rows)
    # Restrict to conditions that have a minimum number of trials
    df = df[df.groupby(conditions)[conditions].transform('size') >= min_trials_per_condition]
    response_matrix_train, condition_matrix_train = df_2_repsonse_matrix(df, conditions)
    for neuron in range(response_matrix_train.shape[3]):
        for trial in range(response_matrix_train.shape[0]):
            for condition in range(response_matrix_train.shape[1]):
                response_matrix_train[trial,condition,:,neuron] = gaussian_filter1d(response_matrix_train[trial,condition,:,neuron]/stride,sigma = sigma)

    return response_matrix_train, condition_matrix_train#, response_matrix_test, condition_matrix_test



# Recurrent neural network model for fitting neural responses





import matplotlib.pyplot as plt
import matplotlib

import torch

plt.rcParams["axes.grid"] = False
matplotlib.rcParams['axes.linewidth'] = .5

import torch.nn.functional as F


# # Recurrent neural network model for fitting neural responses
class RNN(torch.nn.Module):
    def __init__(self, n, input_dim, n_conditions,beta,sigma_rec=0.):
        super(RNN, self).__init__()

        self.n = n
        self.input_dim = input_dim
        self.n_conditions = n_conditions
        self.beta = beta
        self.sigma_rec = sigma_rec
        self.activation = torch.nn.Softplus(beta =self.beta)
        #self.activation = nn.ReLU()
        #self.activation   = LearnableSoftplus(num_features=self.n+self.input_dim, init_beta=1)
        #self.activation = NeuronSpecificSoftplus(self.n+self.input_dim)
        # Hidden dynamics
        self.recurrent_layer = nn.Linear(self.n+self.input_dim, self.n+self.input_dim, bias=True)
        self.recurrent_layer.bias.data.normal_(mean=1, std=0)
        self.recurrent_layer.bias.requires_grad = True

        self.x0_layer =nn.Sequential(nn.Linear(self.n_conditions, self.n+self.input_dim),nn.ReLU(),
                                     nn.Linear(self.n+self.input_dim, self.n+self.input_dim),nn.ReLU(),
                                     nn.Linear(self.n+self.input_dim, self.n+self.input_dim, bias=True))
        #self.x0_layer = nn.Embedding(num_embeddings=3, embedding_dim=self.n+self.input_dim)
        #self.x0_layer.weight.data.normal_(mean=0, std=0)
        #self.x0_layer.bias.data.normal_(mean=0, std=0)
        #self.x0 = torch.nn.Parameter(torch.ones(self.n_conditions,1,self.n+self.input_dim ))
       # self.alpha = torch.nn.Parameter(.1*torch.rand(self.n+self.input_dim))
        self.alpha= .05
        #self.alpha = .01
       # self.observation_layer = nn.Linear(self.n, self.n, bias=True)
#



    # Dynamics
    def forward(self,conditions, steps):
        """
        :param x0:
        :param conditions:
        :param steps:
        :return:
        """
        t = steps

        #states = self.x0_layer(conditions.int()+1)
        states = self.x0_layer(conditions)[:,None,:]
        #states = self.x0
        recurrent_noise = self.sigma_rec * torch.empty(conditions.shape[0], t, self.n+self.input_dim).normal_(
            mean=0,std=1)
       # inputs = []
        for i in range(t - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * (
                self.activation(self.recurrent_layer(states[:, i, :])  ) + states[:, i, :]*recurrent_noise[:,i,:] )
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)

        return states#, torch.stack(inputs)


    # def evaluate(self,conditions,y,steps):
    #     """
    #     :param y:
    #     :param conditions:
    #     :param steps:
    #     :return:
    #     """
    #     # Simulate RNN
    #     x = self.forward(conditions, steps)
    #
    #     # Restrict to non-input neurons
    #     x = x[:,:,:self.n]
    #     # center = torch.mean(x,dim = (0,1))
    #     # scale = torch.std(x,dim = (0,1))
    #     # x_z  = (x - center)/(scale)
    #     #x_z = x
    #     x_z = self.observation_layer(x)
    #
    #    # y_centered = y - torch.mean(y, dim = (0,1), keepdim = True)
    #
    #     # Compute condition averages
    #
    #
    #     # Compute mean squared error for each neuron normalized by the variance in that neuron
    #     score = torch.nn.MSELoss()(x_z, y)
    #     return score








def calculate_spatial_density_correlation(coords, values):
    """
    Calculates correlation between spatial density and point values.

    Parameters:
    coords (np.ndarray): nx2 matrix of (x, y) coordinates
    values (np.ndarray): nx1 matrix of values associated with each point
    """
    # Ensure values is 1D for correlation calculation
    values = values.flatten()

    # 1. Calculate Kernel Density Estimation
    # gaussian_kde expects (2, n) shape, so we transpose coords
    kde = gaussian_kde(coords.T)

    # 2. Evaluate density at each original point
    # This gives us the "spatial density" at the location of every observation
    density = kde.evaluate(coords.T)

    # 3. Calculate Pearson Correlation
    correlation, p_value = pearsonr(density, values)

    return correlation, p_value, density,values