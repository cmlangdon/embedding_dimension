
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

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

from bwm_functions import *


# MAIN LOOP
device = 'cpu'

# Get a list of all items (files and folders) in the directory

directory_path = 'Data'
all_items = os.listdir(directory_path)
print(len(all_items))
# Filter the list to include only folders (directories)
folders_only = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]
#eids = np.load("eids.npy",allow_pickle=True)

for _ in range(100000):

    rows = []
    for _ in range(10):
        try:
            eid = random.choice(folders_only)
           # eid = random.choice(eids)
            full_path_to_random_folder = os.path.join(directory_path, eid)
            print(f"Randomly selected eid: {full_path_to_random_folder}")

           # for _ in range(1):

            dim = 5
            sample_size = 50
            sigma = 5
            latent_dim = dim
            hidden_dim = dim + 2
            n_steps = 100
            lds_lr = .001

            threshold = 1
            min_trials_per_condition  = np.random.choice([10])
            # Check if any folders were found

            # Select a random folder name from the list

           # event = random.choice(["goCueTrigger_times","stimOff_times", "firstMovement_times", "goCue_times", "response_times","feedback_times","stimOn_times"])
            #event = random.choice(["firstMovement_times","feedback_times","stimOn_times"])
            event = np.random.choice(["stimOn_times","firstMovement_times"])
           # conditions = list(np.random.choice(["choice","prev_choice","side","prev_side","reward","prev_reward"],size = np.random.choice([1,2,3]),replace = False))
            conditions = ['contrast']
            #conditions = ["choice","side"]
            # Construct the full path to the randomly selected folder

            clusters = pd.read_pickle(full_path_to_random_folder + '/clusters.pkl')

            clusters['cosmos'] = br.id2acronym(clusters['atlas_id'], mapping="Cosmos")

            spikes = pd.read_pickle(full_path_to_random_folder + '/spikes.pkl')
            trials = pd.read_pickle(full_path_to_random_folder + '/trials.pkl')

          #  responses = get_aligned_response_matrix(spikes, trials[event],stride=.005, step=.005, n_steps=100,sigma = 10)
            counts_series = clusters.groupby('cosmos').size()
            print(counts_series)


            # shape n_pseudo x n_conditions x time x n
            response_matrix_train,condition_matrix_train, response_matrix_test, condition_matrix_test = get_aligned_response_matrix(spikes, clusters, trials, event, conditions, stride=.0025, step=.0025,
                                                        n_steps=n_steps,sigma = sigma, threshold = threshold, min_trials_per_condition = min_trials_per_condition)

            
            print(response_matrix_test.shape)
            # Restrict to good neurons:
            # sub_sample = np.logical_and(np.mean(response_matrix_train, axis=(0, 1, 2)) > threshold,np.mean(response_matrix_test, axis=(0, 1,2)) > threshold)
            # response_matrix_test = response_matrix_test[:, :, :, sub_sample]
            # response_matrix_train = response_matrix_train[:, :, :, sub_sample]

            n_neurons  = response_matrix_train.shape[3]
            print(n_neurons)
            # Choose random 50 neurons from 100 most active
           # neuron_indices = np.argsort(np.mean(response_matrix_train, axis=(0, 1)))[-sample_size:]

           # neuron_indices = np.random.choice(n_neurons, sample_size, replace=False)

            response_matrix_test = response_matrix_test[:, :,:, np.mean(np.std(response_matrix_train, axis=0), axis=(0, 1))>.5]
            response_matrix_train = response_matrix_train[:, :,:, np.mean(np.std(response_matrix_train, axis=0), axis=(0, 1))>.5]
            neuron_indices = np.random.choice(n_neurons, sample_size, replace=False)
            response_matrix_test = response_matrix_test[:, :, :, neuron_indices]
            response_matrix_train = response_matrix_train[:, :, :, neuron_indices]
            # responses_test = responses_test[:, :, neuron_indices]
            # mean_responses_test = mean_responses_test[:, :, neuron_indices]

            # Z-score single neuron responses
            response_matrix_train = (response_matrix_train - np.mean(response_matrix_train, axis=(0, 1, 2), keepdims=True)) / np.std(response_matrix_train, axis=(0, 1, 2), keepdims=True)
            response_matrix_test = (response_matrix_test - np.mean(response_matrix_test, axis=(0, 1, 2), keepdims=True)) / np.std(response_matrix_test, axis=(0, 1, 2), keepdims=True)
        #     responses_test = (responses_test - np.mean(responses_test, axis=(0, 1), keepdims=True)) / np.std(responses_test, axis=(0, 1),
        #                                                                                       keepdims=True)
        #     mean_responses_test = (mean_responses_test - np.mean(mean_responses_test, axis=(0, 1), keepdims=True)) / np.std(
        #         mean_responses_test, axis=(0, 1), keepdims=True)
        # #
            mean_responses_train = np.mean(response_matrix_train, axis=0, keepdims=True)
           # pca_error = compute_pca_error(mean_responses_train.reshape(-1, mean_responses_train.shape[2]), dim)
            pca = PCA().fit(mean_responses_train.reshape(-1, mean_responses_train.shape[2]))
            pca_error =  np.sum(pca.explained_variance_ratio_[dim:])
            pca_1 = np.sum(pca.explained_variance_ratio_[1:])

            if pca_error < 1:
                kmeans_error = compute_kmeans_error(dim, mean_responses_train.reshape(-1, mean_responses_train.shape[2]).T)

                response_matrix_train = torch.tensor(response_matrix_train).float().to(device)
                response_matrix_test = torch.tensor(response_matrix_test).float().to(device)
                # mean_responses = torch.tensor(mean_responses).float().to(device)
                condition_matrix_train = torch.tensor(condition_matrix_train).float().to(device)
                condition_matrix_test = torch.tensor(condition_matrix_test).float().to(device)

                # train_indices = np.random.choice(len(responses), int(.8 * responses.shape[0]), replace=False)
                # test_indices = list(set(range(len(responses))) - set(train_indices))
                # test_indices = train_indices
                y_train = response_matrix_train.reshape(-1,response_matrix_train.shape[2],response_matrix_train.shape[3])
                y_test = response_matrix_test.reshape(-1,response_matrix_train.shape[2],response_matrix_train.shape[3])
                condition_train = condition_matrix_train.reshape(-1,condition_matrix_train.shape[2])
                condition_test = condition_matrix_test.reshape(-1,condition_matrix_train.shape[2])

                # Fit Linear model
                train_losses = []
                test_losses = []
                for fit in range(5):
                    linear_model = LinearNet(n=y_train.shape[2], latent_dim=latent_dim, hidden_dim = hidden_dim, n_conditions = condition_train.shape[1]).to(
                        device)
                    linear_model.fit(steps=y_train.shape[1], epochs=1000, batch_size=100, lr=lds_lr, verbose=True,
                                     weight_decay=0., y_train=y_train,
                                     y_test=y_test,conditions_train = condition_train, conditions_test = condition_test)
                    # linear_test_loss = linear_model.evaluate(condition_test,y_test, y_test.shape[1])
                    #test_linear_loss = linear_model.evaluate( condition_test,y_test, n_steps)
                    #train_linear_loss = linear_model.evaluate(  condition_train,y_train,n_steps)

                    mean_responses = torch.mean(response_matrix_train, dim=0)
                    mean_conditions = torch.mean(condition_matrix_train, dim=0)
                    train_losses.append(linear_model.evaluate(mean_conditions, mean_responses, 100).item())

                    mean_responses = torch.mean(response_matrix_test, dim=0)
                    mean_conditions = torch.mean(condition_matrix_test, dim=0)
                    test_losses.append(linear_model.evaluate(mean_conditions, mean_responses, 100).item())
                linear_error_train_mean = np.mean(train_losses)
                linear_error_train_std = np.std(train_losses)
                linear_error_test_mean = np.mean(test_losses)
                linear_error_test_std = np.std(test_losses)
                #linear_loss = linear_model.evaluate(condition_matrix, responses, n_steps)


                rows.append( {'eid':eid,
                              'n_steps':n_steps,
                              'sample_size':sample_size,
                              'n_neurons':n_neurons,
                              'threshold':threshold,
                              'sigma':sigma,
                              'neurons':neuron_indices,
                              'conditions': conditions,
                                "event":event,
                              "linear_error_train_mean" :linear_error_train_mean,
                            "linear_error_train_std" :linear_error_train_std,
                            "linear_error_test_mean" : linear_error_test_mean,
                            "linear_error_test_std" : linear_error_test_std,
                             'pca_error':pca_error,
                              'pca_1':pca_1,
                             'kmeans_error':kmeans_error,})
        except:
            print("Couldn't find enough neurons")
            continue
    if rows:
        df = pd.DataFrame(rows)
        df.to_pickle("Results/" + ''.join(random.choices(string.ascii_letters + string.digits, k=16)) + '.pkl')
