
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


def restrict_to_region(spikes,cluster_df,beryl):
    within_region_cluster_ids = cluster_df[cluster_df.beryl==beryl].cluster_id.values
    all_cluster_ids = spikes['clusters']
    matching_indices = [index for index, value in enumerate(all_cluster_ids) if value in within_region_cluster_ids]
    spike_clusters = spikes['clusters'][matching_indices]
    spike_times = spikes['times'][matching_indices]
    return spike_clusters, spike_times

def get_aligned_response_matrix(spikes,clusters, trials, event,conditions, stride=.01, step=.01, n_steps=100,sigma = 5,threshold = 5):
    """

    Args:
        spikes: Spike data (time and cluster id)
        event_times: Time (seconds from sessions start) of each event.

    Returns:
        response_matrix : 3D array of shape (n_trials, trial_length, n_neurons)
    """
    # Get total number of trials from event times.
    event_times = trials[event]
    n_trials = len(event_times)

    # Get spike counts in bins aligned to event times.
    intervals = []
    for trial in range(n_trials):
        start = event_times[trial]
        for _ in range (n_steps):
            intervals.append([start, start + stride])
            start += step
    intervals = np.stack(intervals)

   # spike_clusters,spike_times = restrict_to_region(spikes, clusters, beryl)
    spike_count, cluster_id = get_spike_counts_in_bins(spikes['times'], spikes['clusters'],intervals= intervals)

    # Reshape spike_counts in 3D array
    spike_count = spike_count.T.reshape(n_trials,n_steps,-1)


    response_matrix = np.zeros(spike_count.shape)
    for neuron in range(spike_count.shape[2]):
        for trial in range(spike_count.shape[0]):
            response_matrix[trial,:,neuron] = gaussian_filter1d(spike_count[trial,:,neuron]/stride,sigma = sigma)

    # Make response dataframe with trial data

    rows = []
    contrasts = np.nan_to_num(-trials['contrastLeft'], copy=True, nan=1.0) * np.nan_to_num(
    trials['contrastRight'], copy=True, nan=1.0)
    for trial in range(1,n_trials):
        rows.append({'trial':trial,
                     'choice': trials['choice'][trial],
                     'prev_choice': trials['choice'][trial-1],
                     'reward': trials['feedbackType'][trial],
                     'prev_reward': trials['feedbackType'][trial-1],
                     'block': trials['probabilityLeft'][trial],
                     'side': np.sign(contrasts[trial]),
                     'prev_side': np.sign(contrasts[trial-1]),
                     'response': response_matrix[trial,:,:],})


    # Remove inactive neurons
    df = pd.DataFrame(rows)
    response_matrix = []
    condition_matrix = []
    for pseudo_trial in range(2000):

        # Sample a condition
        row = df.sample(n=1)

        # Restrict to trials with that condition
        condition_df = df[df[conditions[0]] == row[conditions[0]].item()]
        for condition in conditions:
            condition_df = condition_df[condition_df[condition] == row[condition].item()]


        # Subsample 5 of those trials
        sub_df = condition_df.sample(frac=1,replace = True)

        # Print("Subsampled df")

        # Compute mean
        mean_df = sub_df.groupby(conditions).response.apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index()
        response_matrix.append(mean_df.response.values[0])
        condition_matrix.append(mean_df[conditions].values[0])
    response_matrix = np.stack(response_matrix,axis=0)
    condition_matrix = np.stack(condition_matrix, axis=0)


    sub_sample = np.mean(response_matrix, axis=(0, 1)) > threshold
    response_matrix = response_matrix[:, :,sub_sample]

    return response_matrix, condition_matrix


# Recurrent neural network model for fitting neural responses
class LinearNet(torch.nn.Module):
    def __init__(self, n,latent_dim,n_conditions):
        super(LinearNet, self).__init__()

        self.n = n
        self.latent_dim = latent_dim
        self.n_conditions =n_conditions
        # Hidden dynamics
        self.recurrent_layer = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim, bias=True))

        self.x0_layer = nn.Sequential(nn.Linear(self.n, self.latent_dim, bias=True))
        self.observation_layer = nn.Sequential(nn.Linear(self.latent_dim, self.n, bias=True))

        self.delay_layer = nn.Sequential(nn.Linear(self.n_conditions, self.latent_dim, bias=True),nn.ReLU())
        self.amplitude_layer = nn.Sequential(nn.Linear(self.n_conditions, self.latent_dim, bias=True))
        self.alpha = torch.nn.Parameter(.01*torch.ones(self.latent_dim))

    def filter(self,u):
        """
        :param t: filter start time
        :return: sparse input
        """
        return torch.nn.ReLU()(torch.exp(-.01 * u**2)-.1)

    # Dynamics
    def forward(self, conditions,y, steps):
        """
        :param x0:
        :param conditions:
        :param steps:
        :return:
        """
        t = steps
        #states = self.x0_layer(y.reshape(y.shape[0],-1))[:,None,:]
      #  states = torch.cat([self.x0_layer(y[:,0,:]),self.u0_layer(conditions)],dim=1)[:,None,:]
        delays = self.delay_layer(conditions)
        amplitudes = self.amplitude_layer(conditions)
        states = self.x0_layer(y[:,None,0,:])
        inputs = []
        for i in range(t - 1):
            input = amplitudes * self.filter(delays-torch.tensor(i).float())
            # state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * self.recurrent_layer(
            #     states[:, i, :] + self.u0_layer(torch.concatenate((conditions,torch.tensor(i).float() * torch.ones((conditions.shape[0],1)).to(device)),dim = 1)))
            state_new = (1 - self.alpha) * states[:, i, :] + self.alpha * self.recurrent_layer(states[:, i, :] + input)
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
            inputs.append(input)
        return states, torch.stack(inputs)


    def evaluate(self,conditions,y,steps):
        """
        :param y:
        :param conditions:
        :param steps:
        :return:
        """
       # x = self.forward(y[:,:self.history_len,:].reshape(y.shape[0],-1), steps)
        x,_ = self.forward(conditions,y, steps)
        x = self.observation_layer(x)
        score = torch.mean(torch.mean((x[:, :, :y.shape[2]] - y) ** 2, dim=[0, 1],
                                     keepdim=True))
        return score

    def fit(self, steps = 100, epochs = 2e6,lr = .001, verbose = True,weight_decay = 0.,batch_size = 128,y_train =None ,y_test = None, conditions_train=None,conditions_test = None,):
        """

        :param steps:
        :param epochs:
        :param lr:
        :param verbose:
        :param weight_decay:
        :param y_train:
        :param conditions_train:
        :return:
        """
        my_dataset = TensorDataset(y_train, conditions_train)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=batch_size,shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        epoch = 0
        while epoch < epochs:

            for batch_idx, (y_batch, conditions_batch) in enumerate(my_dataloader):
                optimizer.zero_grad()
                loss = self.evaluate(conditions_batch,y_batch,steps)
                loss.backward()
                optimizer.step()
                epoch += 1
                if verbose:
                    if epoch % 100 == 0:
                        train_loss = self.evaluate(conditions_train,y_train,steps)
                        val_loss = self.evaluate(conditions_test,y_test, steps)
                        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                        print("loss: {:.4f}".format(train_loss.item()), end=' ')
                        print("val_loss: {:.4f}".format(val_loss.item()))
                        #writer.
                        #add_scalar('Loss/test', loss.item(), epoch)




# Hyperparameters
dim = 5
sigma = 5
latent_dim = 15
n_steps = 100
threshold = 0
directory_path = 'Data'
device = 'cpu'

directory_path = 'Data'
all_items = os.listdir(directory_path)
print(len(all_items))
# Filter the list to include only folders (directories)
folders_only = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]
for run in range(100000):
    results = []
    for _ in range(5):
        df_list = []
        for filename in os.listdir('Results'):
            f = os.path.join('Results', filename)
            if os.path.isfile(f):
                df_list.append(pd.read_pickle(f))
        df = pd.concat(df_list, ignore_index=True)
        df = df[df.n_neurons >= 50]
        df = df[df.pca_error < .1]
        # Sample row
        row  = df.sample(n=1)
        print(row)
        eid = row['eid'].item()
        neuron_indices = row['neurons'].item()
        event = row['event'].item()
        conditions = row['conditions'].item()
        full_path_to_random_folder = os.path.join(directory_path, eid)
        print(f"Randomly selected eid: {full_path_to_random_folder}")


        # Load data for eid
        clusters = pd.read_pickle(full_path_to_random_folder + '/clusters.pkl')
        spikes = pd.read_pickle(full_path_to_random_folder + '/spikes.pkl')
        trials = pd.read_pickle(full_path_to_random_folder + '/trials.pkl')
        n_trials = len(trials['choice'])
        # Get mean responses
        try:
            response_matrix, condition_matrix = get_aligned_response_matrix(spikes,
                                                                                                             clusters,
                                                                                                             trials, event,
                                                                                                             conditions,
                                                                                                             stride=.005,
                                                                                                             step=.005,
                                                                                                             n_steps=n_steps,
                                                                                                             sigma=sigma,
                                                                                                             threshold=threshold)
        except:
            print("Failed to get response matrix")
            continue


        # Restrict responses to neuron indices
        print(response_matrix.shape)
        responses = response_matrix[:, :, neuron_indices]
      #  responses = responses[:, :, neuron_indices]
       # print(responses.shape[0])
        # Preprocess for linear fitting
        responses = torch.tensor(responses).float().to(device)
       # mean_responses = torch.tensor(mean_responses).float().to(device)
        condition_matrix = torch.tensor(condition_matrix).float().to(device)
       # mean_condition_matrix = torch.tensor(mean_condition_matrix).float().to(device)
        responses = (responses - torch.mean(responses, dim=[0, 1], keepdim=True)) / torch.std(responses, dim=[0, 1],
                                                                                          keepdim=True)
        # mean_responses = (mean_responses - torch.mean(mean_responses, dim=[0, 1], keepdim=True)) / torch.std(
        #     mean_responses, dim=[0, 1], keepdim=True)

        # Train test split
        train_indices = np.random.choice(len(responses), int(.8 * responses.shape[0]), replace=False)
        test_indices = list(set(range(len(responses))) - set(train_indices))
       # test_indices = train_indices
        y_train = responses[train_indices]
        y_test = responses[test_indices]
        condition_train = condition_matrix[train_indices]
        condition_test = condition_matrix[test_indices]


        #Fit Linear model
        linear_model = LinearNet(n=responses.shape[2], latent_dim=latent_dim,n_conditions = condition_matrix.shape[1]).to(
             device)
        linear_model.fit(steps=responses.shape[1], epochs=1000,batch_size = 100, lr=.001, verbose=True, weight_decay=0., y_train=y_train,
                   y_test=y_test,conditions_train = condition_train,conditions_test = condition_test  )
        # linear_test_loss = linear_model.evaluate(condition_test,y_test, y_test.shape[1])
        test_linear_loss = linear_model.evaluate(condition_test, y_test, n_steps)
        train_linear_loss = linear_model.evaluate(condition_train, y_train, n_steps)
        linear_loss = linear_model.evaluate(condition_matrix,responses,n_steps)
        print(linear_loss.item())
        results.append( {'eid':eid,
                  'neuron_indices':neuron_indices,
                  'event':event,
                  'conditions':conditions,
                    'n_trials':n_trials,
                  'pca_error':row['pca_error'].item(),
                  'kmeans_error':row['kmeans_error'].item(),
                  'linear_error':linear_loss.item(),
                  'test_linear_error':test_linear_loss.item(),
                  'train_linear_error':train_linear_loss.item(),})

    pd.DataFrame(results).to_pickle("LinearResults/" + ''.join(random.choices(string.ascii_letters + string.digits, k=16)) + '.pkl')

