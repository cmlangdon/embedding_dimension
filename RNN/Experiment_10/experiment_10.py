"""
Train networks on 2AFC task with penalty for variance along higher principal components.
"""
from net import *
import torch
import random 
import string
import os
import itertools
import numpy as np

import pandas as pd
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans


task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
#task_id = 1


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)




def compute_pca_error(responses, dim):
    pca = PCA().fit(responses.T)
    return np.sum(pca.explained_variance_ratio_[dim:])


# Function for computing r2 for kmeans model
def compute_kmeans_error(dim, responses):
    kmeans = KMeans(n_clusters=dim, init='random', n_init=100).fit(responses)

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

def iterative_outlier_detection(x):
    outliers_found = True
    while outliers_found:
        center = np.mean(np.mean(x, axis=(0,1)))
        dev = np.std(np.mean(x, axis=(0,1)))
        scores = np.abs((np.mean(x, axis=(0,1)) - center) / dev)
        x = x[:,:,scores<2]
        if np.sum(scores>2)==0:
            outliers_found = False
    return x

def cayley_transform(a):
    N = a.shape[0]
    # Tranform square matrix, a, into orthonormal matrix, q.
    skew = (a - a.t()) / 2
    skew = skew
    eye = torch.eye(N).to()
    o = (eye - skew) @ torch.inverse(eye + skew)
    return o

def null_data(responses):

    responses = torch.tensor(responses).float()
    a = torch.nn.Parameter(10 * (2 * torch.rand(responses.shape[0], responses.shape[0]) - 1), requires_grad=True)
    q = cayley_transform(a)
    rot_responses = q @ responses

    return rot_responses.detach().numpy()

# Set parameter grid
from ParametricWMTask import generate_trials

u, z, mask, conditions = generate_trials(n_trials=25)


# 
# n_init = 5
n_init = 5

# beta = [10]
max_k = 5
n_runs = 1
n_repeats = 1




# for job in job_intervals[task_id - 1]:
#     print(job)
for _ in range(100):
    print('batch: {}'.format(_))
    results = []
    for run in range(5):
        print('run: {}'.format(run))
        dim = 4
        params = {'n_neurons': 100,
                  'epochs': 1500,
                  'dim': dim,
                  'sigma_rec': 0,
                  'weight_decay': .001,
                  'lambda_std': .4,
                  'sigma_in': 0.,
                  'base_in': 0,
                  'lr': .005,
                  'beta': 50,
                  'pca_target_2':  np.random.choice(np.linspace(0,.5,100)),
                  'gain_target': np.random.choice(np.linspace(0,.5,100)),
                  'lambda_gain_mean': 1,
                  'lambda_gain_std': 1,
                  'lambda_pca_1': np.random.choice(np.linspace(0.2,.5,10)),
                  'lambda_pca_2': 1,
                  }




        net = Net(n=params['n_neurons'], input_size=u.shape[2],output_size=u.shape[2],
                  dale=False,
                  sigma_in=params['sigma_in'],
                  base_in=params['base_in'],
                  lambda_std=params['lambda_std'],
                  beta=params['beta']).to(device)
    
        # Fit with penalty
        net.fit(u.to(device=device),
                z.to(device=device),
                mask.to(device=device),
                lr=params['lr'],
                epochs=params['epochs'],
                conditions=conditions,
                verbose=True,
                dim=params['dim'],
                weight_decay=params['weight_decay'],
                pca_target_2=params['pca_target_2'],
                gain_target=params['gain_target'],
                lambda_pca_1=params['lambda_pca_1'],
                lambda_pca_2=params['lambda_pca_2'],
                lambda_gain_mean=params['lambda_gain_mean'],
                lambda_gain_std=params['lambda_gain_std'],
                      )
    
        # Performance metrics with no recurrent noise.
        net.sigma_in = 0.0
        net.sigma_rec = 0.0
        x = net(u.to(device=device))

    
        # Condition averages
        x = x.detach().cpu().numpy()
       # x = iterative_outlier_detection(x)
        rows = []
        for k in range(u.shape[0]):
            rows.append({'trial': k,
                         'theta': conditions[k]['theta'],
                         'response': x[k, :, :]})
        df = pd.DataFrame(rows)
        df = df.groupby('theta').response.apply(lambda r: np.mean(np.stack(r), axis=0)).reset_index()
        responses = np.stack(df.response.values)
        responses = responses.reshape(-1, responses.shape[2]).T
    
        # z_score
        responses = (responses - np.mean(responses, axis=1, keepdims=True)) / np.std(responses, axis=1, keepdims=True)
        responses = responses[~np.isnan(responses).any(axis=1)]
    
        # PCA error
        pca_error = compute_pca_error(responses, params['dim'])
        pca = PCA().fit(responses.T)

        # Parallelism metrics
        x = torch.tensor(x).float().to(device)
        gain_mean, gain_std = net.plane_parallelism(x,u)
       # non_linearity_2 = net.flow_parallelism(x, u).item()

        # K Means error
        kmeans_error = compute_kmeans_error(params['dim'], responses)




        results.append({
            'model_id': ''.join(random.choices(string.ascii_letters + string.digits, k=8)),
            'epochs': params['epochs'],
            'sigma_rec': params['sigma_rec'],
            'sigma_in': params['sigma_in'],
            'weight_decay': params['weight_decay'],
            'beta': params['beta'],
            'lr': params['lr'],
            'lambda_std': params['lambda_std'],
            'pca_target_2' : params['pca_target_2'],
            'lambda_pca_1' : params['lambda_pca_1'],
            'lambda_pca_2' : params['lambda_pca_2'],
            'gain_mean' : gain_mean.item(),
            'gain_std': gain_std.item(),
            'n': responses.shape[0],
            'dim': params['dim'],
            'pca_error': pca_error,
            'pca_1': pca.explained_variance_ratio_[dim-1],
            'kmeans_error': kmeans_error,
            'lambda_gain_mean': params['lambda_gain_mean'],
            'lambda_gain_std': params['lambda_gain_std'],
    
        })

    # except:
    #     print('Failed on job')
    #     continue

# df = pd.DataFrame(results)
# df.to_pickle("Results_10/results_" + str(task_id ) + ".pkl")
    df = pd.DataFrame(results)
    df.to_pickle("Results_10/" + ''.join(random.choices(string.ascii_letters + string.digits, k=16)) + '.pkl')