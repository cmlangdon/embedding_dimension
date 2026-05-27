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
        x = x[:,:,scores<3]
        if np.sum(scores>3)==0:
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

u, z, mask, conditions = generate_trials(n_trials=5)

lvar = [2]

dim = [4]
threshold = [.05]
n_neurons = [100]  # Changed
epochs = [100]
lr = [.005]
# 0.2
kmeans_lambda  = [0]
pca_lambda =  [0]
linear_lambda =   [0]

kmeans_target = [-.01,0,.01]
pca_target = [-1,0,1]
linear_target = [-.01,0,.01]
n_init = 5
sigma_in = [0.]
base_in = [0.2]
sigma_rec = [0.]
weight_decay = [0.,.001,.002,.004]
lambda_std = [.2,.4,1]  # Changed
beta = [10]
max_k = 5
n_runs = 1
n_repeats = 4
param_grid = np.repeat(
    np.array([x for x in
              itertools.product(n_neurons, epochs, lvar, dim, sigma_rec, weight_decay, threshold, beta, lambda_std,
                                sigma_in, base_in, lr, beta,kmeans_target,pca_target,linear_target,kmeans_lambda,pca_lambda,linear_lambda)]),
    repeats=n_repeats, axis=0)

# Loop through parameters
n_jobs = param_grid.shape[0]
print(n_jobs)
n_cluster_jobs = 1296
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
                  'lambda_std': (param_grid[job][8]).astype(float),
                  'sigma_in': (param_grid[job][9]).astype(float),
                  'base_in': (param_grid[job][10]).astype(float),
                  'lr': (param_grid[job][11]).astype(float),
                  'beta': (param_grid[job][12]).astype(float),
                  'kmeans_target': (param_grid[job][13]).astype(float),
                  'pca_target': (param_grid[job][14]).astype(float),
                  'linear_target': (param_grid[job][15]).astype(float),
                  'kmeans_lambda': (param_grid[job][16]).astype(float),
                  'pca_lambda': (param_grid[job][17]).astype(float),
                  'linear_lambda': (param_grid[job][18]).astype(float),
                  }

        net = Net(n=params['n_neurons'], input_size=u.shape[2],
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
                pca_target=params['pca_target'],
                linear_target=params['linear_target'],

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

        # # LDS error
        lds = LDS(n=params['dim'],output_size = x.shape[2]).to(device=device)
       # x = net.forward(u.to(device=device))
        x = torch.tensor(x).float().to(device)

        lds.fit(u.to(device=device), x.detach(), epochs=1000)
        y = lds.output_layer(lds(u.to(device=device)))
        linear_error = net.linear_error(x,y).detach().item()

        kmeans_error = compute_kmeans_error(params['dim'], responses)


        # Null test
        null_kmeans_errors = []
        for run in range(n_runs):
            rot_responses = null_data(responses)
            null_kmeans_errors.append(compute_kmeans_error(params['dim'], rot_responses))
        null_kmeans_error = np.mean(null_kmeans_errors)

        results.append({
            'model_id': ''.join(rdm.choices(string.ascii_letters + string.digits, k=8)),
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
            'kmeans_target': params['kmeans_target'],
            'pca_target': params['pca_target'],
            'linear_target': params['linear_target'],
            'n': responses.shape[0],
            'lvar': params['lvar'],
            'dim': params['dim'],
            'null_kmeans_error': null_kmeans_error,
            'pca_error': pca_error,
            'kmeans_error': kmeans_error,
            'linear_error': linear_error,

        })

    except:
        print('Failed on job')
        continue

df = pd.DataFrame(results)
df.to_pickle("Results_4/results_" + str(task_id ) + ".pkl")
