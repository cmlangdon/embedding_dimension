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
from scipy.sparse import save_npz, load_npz, csr_matrix

# task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

from bwm_functions import *

# MAIN LOOP
device = 'cpu'

# Get a list of all items (files and folders) in the directory

directory_path = 'Data'
all_items = os.listdir(directory_path)
print(len(all_items))
# Filter the list to include only folders (directories)
folders_only = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]
# eids = np.load("eids.npy",allow_pickle=True)

for _ in range(100000):

    rows = []
    for _ in range(5):
        try:

            # Select random eid
            eid = random.choice(folders_only)
            # eid = random.choice(eids)
            full_path_to_random_folder = os.path.join(directory_path, eid)
            print(f"Randomly selected eid: {full_path_to_random_folder}")
            # Hyperparameters
            dim = 10
            sample_size = 50
            sigma = 10
            n_steps = 100
            threshold = .01

            min_trials_per_condition = np.random.choice([100])
            # Check if any folders were found

            # Select a random folder name from the list

            event = random.choice(["firstMovement_times", "feedback_times", "stimOn_times", "response_times"])
            conditions = list(np.random.choice(["choice", "prev_choice", "side", "prev_side", "reward", "prev_reward"],
                                               size=np.random.choice([1, 2, 3]), replace=False))

            clusters = pd.read_pickle(full_path_to_random_folder + '/clusters.pkl')

            clusters['cosmos'] = br.id2acronym(clusters['atlas_id'], mapping="Cosmos")

            spikes = pd.read_pickle(full_path_to_random_folder + '/spikes.pkl')
            trials = pd.read_pickle(full_path_to_random_folder + '/trials.pkl')

            counts_series = clusters.groupby('cosmos').size()
            print(counts_series)

            # shape n_pseudo x n_conditions x time x n
            response_matrix_train, condition_matrix_train, response_matrix_test, condition_matrix_test = get_aligned_response_matrix(
                spikes, clusters, trials, event, conditions,
                stride=.005,
                step=.005,
                n_steps=n_steps,
                sigma=sigma,
                threshold=threshold,
                min_trials_per_condition=min_trials_per_condition)

            # Restrict to active units
            restriction = np.logical_and(np.std(response_matrix_train, axis=(0, 1, 2)) > threshold,
                                         np.std(response_matrix_test, axis=(0, 1, 2)) > threshold)
            response_matrix_train = response_matrix_train[
                :, :, :, restriction]
            response_matrix_test = response_matrix_test[
                :, :, :, restriction]

            # Restrict to least noisy neurons
            # response_matrix_test = response_matrix_test[
            #     :, :, :, np.mean(np.std(response_matrix_train, axis=0), axis=(0, 1)) > .5]
            # response_matrix_train = response_matrix_train[
            #     :, :, :, np.mean(np.std(response_matrix_train, axis=0), axis=(0, 1)) > .5]

            print(response_matrix_train.shape)

            # Randomly sample from these neurons
            n_neurons = response_matrix_train.shape[3]
            neuron_indices = np.random.choice(n_neurons, sample_size, replace=False)
            response_matrix_train = response_matrix_train[:, :, :, neuron_indices]
            response_matrix_test = response_matrix_test[:, :, :, neuron_indices]

            # Calculate mean snr for neurons

            n_conditions = response_matrix_train.shape[1]
            # Z-score single neuron responses
            response_matrix_train = (response_matrix_train - np.mean(response_matrix_train, axis=(0, 1, 2),
                                                                     keepdims=True)) / np.std(response_matrix_train,
                                                                                              axis=(0, 1, 2),
                                                                                              keepdims=True)
            response_matrix_test = (response_matrix_test - np.mean(response_matrix_test, axis=(0, 1, 2),
                                                                   keepdims=True)) / np.std(response_matrix_test,
                                                                                            axis=(0, 1, 2),
                                                                                            keepdims=True)

            mean_responses_train = np.mean(response_matrix_train, axis=0, keepdims=True)
            pca = PCA().fit(mean_responses_train.reshape(-1, mean_responses_train.shape[2]))
            pca_error = np.sum(pca.explained_variance_ratio_[dim:])
            pca_1 = pca.explained_variance_ratio_[1]

            # if pca_error < 1:
            kmeans_error = compute_kmeans_error(dim,
                                                mean_responses_train.reshape(-1, mean_responses_train.shape[2]).T)

            ##################################################
            ############## Fit RNN and compute gain ##########
            ##################################################
            # Preprocess data
            response_matrix_train = torch.tensor(response_matrix_train).float().to(device)
            response_matrix_test = torch.tensor(response_matrix_test).float().to(device)
            condition_matrix_train = torch.tensor(condition_matrix_train).float().to(device)
            condition_matrix_test = torch.tensor(condition_matrix_test).float().to(device)
            y_train = response_matrix_train.reshape(-1, response_matrix_train.shape[2],
                                                    response_matrix_train.shape[3])
            y_test = response_matrix_test.reshape(-1, response_matrix_train.shape[2],
                                                  response_matrix_train.shape[3])
            condition_train = condition_matrix_train.reshape(-1, condition_matrix_train.shape[2])
            condition_test = condition_matrix_test.reshape(-1, condition_matrix_train.shape[2])

            # Initialize RNN and fit
            rnn_model = RNN(n=y_train.shape[2], input_dim=2, latent_dim=5,
                            n_conditions=condition_train.shape[1]).to(
                device)
            rnn_model.fit(steps=y_train.shape[1], epochs=1000, batch_size=100, lr=.005, verbose=True,
                          weight_decay=0., y_train=y_train,
                          y_test=y_test, conditions_train=condition_train, conditions_test=condition_test)



            linear_test_loss = rnn_model.evaluate(condition_test,y_test, y_test.shape[1])
            test_linear_loss = rnn_model.evaluate( condition_test,y_test, n_steps)
            train_linear_loss = rnn_model.evaluate(  condition_train,y_train,n_steps)

            mean_responses = torch.mean(response_matrix_train, dim=0)
            mean_conditions = torch.mean(condition_matrix_train, dim=0)
            # train_losses.append(rnn_model.evaluate(mean_conditions, mean_responses, 100).item())

            x = rnn_model.forward(mean_conditions, steps=100)

            h = rnn_model.recurrent_layer(x)
            h = h - torch.mean(h, dim=(0, 1), keepdim=True)

            z = x - torch.mean(x, dim=[0, 1], keepdim=True)

            beta = torch.sum(h * z, dim=(0, 1)) / torch.sum(z ** 2, dim=(0, 1))

            gain_mean = torch.mean(torch.abs(beta[:rnn_model.n]))
            gain_std = torch.std(torch.abs(beta[:rnn_model.n]))

            rows.append({'eid': eid,
                         'n_steps': n_steps,
                         'sample_size': sample_size,
                         'n_neurons': n_neurons,
                         'threshold': threshold,
                         'sigma': sigma,
                         'neurons': neuron_indices,
                         'conditions': conditions,
                         'n_conditions': n_conditions,
                         "event": event,
                         'pca_error': pca_error,
                         'pca_1': pca_1,
                         'kmeans_error': kmeans_error,
                         'gain_mean': gain_mean.item(),
                         'gain_std': gain_std.item(),})
        except:
            print("Couldn't find enough neurons")
            continue
    if rows:
        df = pd.DataFrame(rows)
        df.to_pickle("Results/" + ''.join(random.choices(string.ascii_letters + string.digits, k=16)) + '.pkl')
