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


#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
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
    kmeans = KMeans(n_clusters=dim, init='random', n_init=10).fit(responses)

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

# lvar = [2]
# 
# dim = [3]
# threshold = [.05]
# n_neurons = [100]  # Changed
# epochs = [1000]
# lr = np.linspace(.001,.01,100)

# lambda_kmeans = [0]
# lambda_pca_1 = [1]
# lambda_pca_2 = [1]
# lambda_linear =  [0]
# kmeans_target = [.5]
# pca_target_1 = np.linspace(0,.33,25)
# pca_target_2 = np.linspace(0,.25,25)
# linear_target = [.5]
# 
# n_init = 5
n_init = 5
# sigma_in = [0.]
# base_in = [0.2]
# sigma_rec = [0.]
# weight_decay = [0]
# lambda_std = [.4]  # Changed
# 
# beta = [10]
max_k = 5
n_runs = 1
n_repeats = 1


# param_grid = np.repeat(
#     np.array([x for x in
#               itertools.product(n_neurons, epochs, lvar, dim, sigma_rec, weight_decay, threshold, beta, lambda_std,
#                                 sigma_in, base_in, lr, beta,kmeans_target,pca_target_1,pca_target_2,linear_target,lambda_kmeans,lambda_linear,lambda_pca_1,lambda_pca_2)]),
#     repeats=n_repeats, axis=0)

# Loop through parameters
#n_jobs = param_grid.shape[0]
#print(n_jobs)
#n_cluster_jobs = 40
#job_intervals = np.split(np.arange(n_jobs), n_cluster_jobs)

# for job in job_intervals[task_id - 1]:
#     print(job)
for _ in range(100):
    print('batch: {}'.format(_))
    results = []
    for run in range(5):
        print('run: {}'.format(run))
        dim = 3
        params = {'n_neurons': 100,
                  'epochs': 500,
                  'lvar': 1,
                  'dim': dim,
                  'sigma_rec': 0,
                  'weight_decay': .001,
                  'threshold': .05,
                  'lambda_std': .4,
                  'sigma_in': 0.,
                  'base_in': 0,
                  'lr': .005,
                  'beta': 10.,
                  'kmeans_target': np.random.choice(np.linspace(0,.75,25)),
                  'pca_target_1':  1,
                  'pca_target_2':  np.random.choice(np.linspace(0,1,25)),
                  'linear_target': np.random.choice(np.linspace(0,.75,25)),
                  'lambda_kmeans': 1,
                  'lambda_linear': 1,
                  'lambda_pca_1': .25,
                  'lambda_pca_2': 1
                  }
        
    
        net = Net(n=params['n_neurons'], input_size=u.shape[2],output_size=u.shape[2],
                  dale=False,
                  sigma_in=params['sigma_in'],
                  base_in=params['base_in'],
                  lambda_std=params['lambda_std'],
                  beta=params['beta']).to(device)
    
        # Fit with penalty
        lds = net.fit(u.to(device=device),
                z.to(device=device),
                mask.to(device=device),
                lr=params['lr'],
                epochs=params['epochs'],
                conditions=conditions,
                verbose=True,
                lvar=params['lvar'],
                dim=params['dim'],
                weight_decay=params['weight_decay'],
                kmeans_target=params['kmeans_target'],
                pca_target_1=params['pca_target_1'],
                pca_target_2=params['pca_target_2'],
                linear_target=params['linear_target'],
                lambda_kmeans=params['lambda_kmeans'],
                lambda_pca_1=params['lambda_pca_1'],
                lambda_pca_2=params['lambda_pca_2'],
                lambda_linear=params['lambda_linear'],
    
                      )
    
        # Performance metrics with no recurrent noise.
        net.sigma_in = 0.0
        net.sigma_rec = 0.0
        x = net(u.to(device=device))
        mse_z = net.mse_z(x.to(device), z.to(device), mask.to(device))
    
    
        # Condition averages
    
        x = x.detach().cpu().numpy()
        x = iterative_outlier_detection(x)
    
        # snr
        snr = np.std(np.std(x, axis=(0, 1)))/np.mean(np.std(x, axis=(0, 1)))
    
        rows = []
        for k in range(u.shape[0]):
            rows.append({'trial': k,
                         'theta': conditions[k]['theta'],
                         'response': x[k, :, :]})
        df = pd.DataFrame(rows)
        df = df.groupby('theta').response.apply(lambda r: np.mean(np.stack(r), axis=0)).reset_index()
        responses = np.stack(df.response.values)
        responses = responses.reshape(-1, responses.shape[2]).T
        #responses = responses[np.mean(responses, axis=1) > .025, :]
    
        # z_score
        responses = (responses - np.mean(responses, axis=1, keepdims=True)) / np.std(responses, axis=1, keepdims=True)
        responses = responses[~np.isnan(responses).any(axis=1)]
    
        # PCA error
        pca_error = compute_pca_error(responses, params['dim'])
    
        x = torch.tensor(x).float().to(device)
        x = (x - torch.mean(x, dim=(0, 1))) / torch.std(x, dim=(0, 1))
        linear_errors = []
        for _ in range(5):
            lds = LDS(n=dim, input_size = 2,output_size=x.shape[2]).to(device=device)
            lds.fit(u.to(device=device), x, epochs=2000,lr = .01)
            y = lds.output_layer(lds(u.to(device=device)))
            linear_error = net.linear_error(y, x).detach().item()
            linear_errors.append(linear_error)
        linear_error_mean = np.mean(linear_errors)
        linear_error_std = np.std(linear_errors)
    
        kmeans_error = compute_kmeans_error(params['dim'], responses)
    
    
        # # Null test
        # null_kmeans_errors = []
        # for run in range(n_runs):
        #     rot_responses = null_data(responses)
        #     null_kmeans_errors.append(compute_kmeans_error(params['dim'], rot_responses))
        # null_kmeans_error = np.mean(null_kmeans_errors)
        #
        results.append({
            'model_id': ''.join(random.choices(string.ascii_letters + string.digits, k=8)),
            'w_rec': net.recurrent_layer.weight.data.detach().cpu().numpy(),
            'w_in': net.input_layer.weight.data.detach().cpu().numpy(),
            'w_out': net.output_layer.weight.data.detach().cpu().numpy(),
            'bias': net.recurrent_layer.bias.data.detach().cpu().numpy(),
            'mse_z': mse_z.item(),
            'epochs': params['epochs'],
            'sigma_rec': params['sigma_rec'],
            'sigma_in': params['sigma_in'],
            'weight_decay': params['weight_decay'],
            'snr':snr,
            'beta': params['beta'],
            'lr': params['lr'],
            'lambda_std': params['lambda_std'],
            'kmeans_target' : params['kmeans_target'],
            'pca_target_1' : params['pca_target_1'],
            'pca_target_2' : params['pca_target_2'],
            'linear_target' : params['linear_target'],
            'lambda_kmeans' : params['lambda_kmeans'],
            'lambda_pca_1' : params['lambda_pca_1'],
            'lambda_pca_2' : params['lambda_pca_2'],
            'lambda_linear' : params['lambda_linear'],
            'n': responses.shape[0],
            'lvar': params['lvar'],
            'dim': params['dim'],
            'pca_error': pca_error,
            'kmeans_error': kmeans_error,
            'linear_error_mean': linear_error_mean,
            'linear_error_std': linear_error_std,
    
        })

    # except:
    #     print('Failed on job')
    #     continue

# df = pd.DataFrame(results)
# df.to_pickle("Results_10/results_" + str(task_id ) + ".pkl")
    df = pd.DataFrame(results)
    df.to_pickle("Results_11/" + ''.join(random.choices(string.ascii_letters + string.digits, k=16)) + '.pkl')