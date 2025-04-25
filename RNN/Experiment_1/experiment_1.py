"""
Train networks on 2AFC task with penalty for variance along higher principal components.
"""
from net import *
import torch
import random as rdm
import string
import os
import itertools
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from scipy.stats import ortho_group
from sklearn.metrics import calinski_harabasz_score

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from scipy import stats
# from datajoint_tables import *
# from datajoint_tables import *
from sklearn.metrics.cluster import adjusted_rand_score

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 1

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)


# Function for computing number of clusters
def compute_k(responses, max_k, threshold):
    inertia = []
    for n_clusters in range(1, np.min([max_k, responses.shape[1]]) + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init=5).fit(responses)
        inertia.append(kmeans_r2(kmeans, responses))
    # inertia = -np.diff(inertia)
    for j in range(len(inertia)):
        if inertia[j] < threshold:
            break
    k = j + 1
    return inertia, k


# Function for computing r2 for kmeans model
def kmeans_r2(kmeans, responses):
    total_variance = np.sum((responses.T - np.mean(responses.T, axis=0, keepdims=True)) ** 2)
    return kmeans.inertia_ / total_variance


# Set parameter grid
from TwoAFCTask import generate_trials

u, z, mask, conditions = generate_trials(n_trials=25)

lvar = np.concatenate([-10 ** np.linspace(-3, 0, 25), [0], 10 ** np.linspace(-3, 0, 25)])
lvar = [1]
dim = [2]
threshold = [.025]
n_neurons = [100]
epochs = [1000]
lr = [.005]
n_init = 5
sigma_rec = [0.]
lambda_std = [0.1]
weight_decay = [0.001]
max_k = 6
n_runs = 5
n_repeats = 50
param_grid = np.repeat(
    np.array([x for x in itertools.product(n_neurons, epochs, lvar, dim, sigma_rec, weight_decay, threshold,lambda_std,lr)]),
    repeats=n_repeats, axis=0)

# Loop through parameters
n_jobs = param_grid.shape[0]
n_cluster_jobs = 25
job_intervals = np.split(np.arange(n_jobs), n_cluster_jobs)
results = []
for job in job_intervals[task_id - 1]:
    print(job)
    try:
        params = {'n_neurons': (param_grid[job][0]).astype(int),
                  'epochs': (param_grid[job][1]).astype(int),
                  'lvar': (param_grid[job][2]).astype(float),
                  'dim': (param_grid[job][3]).astype(int),
                  'sigma_rec': (param_grid[job][4]).astype(float),
                  'weight_decay': (param_grid[job][5]).astype(float),
                  'threshold': (param_grid[job][6]).astype(float),
                  'lambda_std': (param_grid[job][7]).astype(float),
                  'lr': (param_grid[job][8]).astype(float),
                  }
        
        # Initialize net
        net = Net(n=params['n_neurons'], 
                  input_size=u.shape[2], 
                  dale=False, 
                  sigma_in=0.1,
                  sigma_rec=params['sigma_rec'],
                  lambda_std=params['lambda_std']).to(device)
    
        # Fit with penalty
        net.fit(u.to(device=device), z.to(device=device), mask.to(device=device), 
                lr=params['lr'],
                epochs=params['epochs'],
                conditions=conditions,
                verbose=True,
                lvar=params['lvar'],
                dim=params['dim'],
                weight_decay=params['weight_decay'])
    
    
        # Performance metrics
        net.sigma_rec = 0.0
        x = net(u.to(device=device))
        mse_z = net.mse_z(x.to(device), z.to(device), mask.to(device))
        activity_std = torch.std(torch.std(x, dim=[0, 1])).detach().cpu().numpy()
    
        # Condition averages
        x = x.detach().cpu().numpy()
        rows = []
        for k in range(u.shape[0]):
            rows.append({'trial': k,
                         'motion': conditions[k]['motion_coh'],
                         'response': x[k, :, :]})
        df = pd.DataFrame(rows)
        df = df.groupby('motion').response.apply(lambda r: np.mean(np.stack(r), axis=0)).reset_index()
        responses = np.stack(df.response.values)
        responses = responses.reshape(-1, responses.shape[2]).T
    
        # Remove inactive neurons
        responses = responses[np.mean(responses, axis=1) > 0.025, :]
        
        # z_score
        responses = (responses - np.mean(responses, axis=1, keepdims=True)) / np.std(responses, axis=1, keepdims=True)
        responses = responses[~np.isnan(responses).any(axis=1)]
    
        # Compute variance explained ratios
        variance = PCA().fit(responses.T).explained_variance_ratio_
    
        # Compute k
        inertia, k = compute_k(responses, max_k, params['threshold'])
    
        # Null test
        null_inertia = np.zeros((max_k, n_runs))
        for run in range(n_runs):
            # random_responses = np.random.multivariate_normal(mean, cov, responses.shape[1]).T
            rot = ortho_group.rvs(responses.shape[0])
            rot_responses = rot @ responses
            null_inertia[:, run], _ = compute_k(rot_responses, max_k, threshold)
    
        p_value = np.sum(null_inertia[k - 1, :] < inertia[k - 1]) / n_runs
        print(responses.shape[0])
        results.append({
            'model_id': ''.join(rdm.choices(string.ascii_letters + string.digits, k=8)),
            'w_rec': net.recurrent_layer.weight.data.detach().cpu().numpy(),
            'w_in': net.input_layer.weight.data.detach().cpu().numpy(),
            'w_out': net.output_layer.weight.data.detach().cpu().numpy(),
            'bias': net.recurrent_layer.bias.data.detach().cpu().numpy(),
            'mse_z': mse_z.item(),
            'weight_decay': params['weight_decay'],
            'threshold': params['threshold'],
            'sigma_rec': params['sigma_rec'],
            'lambda_std': params['lambda_std'],
            'lr':params['lr'],
            'n': responses.shape[0],
            'lvar': params['lvar'],
            'dim': params['dim'],
            'k': k,
            'p_value': p_value,
            'inertia': inertia,
            'activity_std':activity_std,
            'null_inertia': null_inertia,
            'variance': variance,
        })

    except:
        print('Failed on job')
        continue

df = pd.DataFrame(results)
df.to_pickle("Results_1/results_" + str(task_id) + ".pkl")
