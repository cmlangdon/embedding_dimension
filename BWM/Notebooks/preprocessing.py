import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import datajoint as dj_langdon
dj_langdon.config['database.host'] = 'datajoint-tengel.pni.princeton.edu'
dj_langdon.config['database.user'] = 'cl1704'
dj_langdon.config['database.password'] = 'wuxty2-mYdxej-kerxaq'
schema = dj_langdon.schema('langdon_ibl_manifold')
schema.spawn_missing_classes()
from sklearn.cluster import KMeans
#from kneed import KneeLocator
from scipy.stats import ks_2samp
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score


def session_data(eid):
    data = (Session() & {'eid': eid}).fetch1()
    stimon_times = data['stimon_times']
    choices = data['choice']
    contrasts = np.nan_to_num(-data['contrast_left'], copy=True, nan=1.0) * np.nan_to_num(
        data['contrast_right'], copy=True, nan=1.0)
    contrasts[contrasts == 0.] = 0
    sides = np.sign(contrasts)
    p_left = data['probability_left']
    previous_choice = np.roll(choices,1)
    return stimon_times, choices, sides




def binned_trials(spike_times,stimon_times,choices,sides,n_time, trial_length):
    statistics = []
    p_values = []
    for neuron in range(spike_times.shape[0]):
        neuron_spikes = spike_times[neuron]
        pre_spikes = []
        post_spikes = []
        for trial in range(1, len(stimon_times)):
            # Align to stimon
            pre_trial_spikes = neuron_spikes[np.logical_and(neuron_spikes > stimon_times[trial] - trial_length,
                                                            neuron_spikes < stimon_times[trial])]
            pre_spikes.append(len(pre_trial_spikes))

            post_trial_spikes = neuron_spikes[np.logical_and(neuron_spikes > stimon_times[trial],
                                                             neuron_spikes < stimon_times[trial] + trial_length)]
            post_spikes.append(len(post_trial_spikes))

        try:
            statistic, p = ks_2samp(pre_spikes, post_spikes)
            statistics.append(statistic)
            p_values.append(p)
        except:
            statistics.append(0.0)
            p_values.append(1.0)

    statistics = np.array(statistics)
    p_values = np.array(p_values)

    good_units = np.argwhere(np.array(p_values) < .05)[:, 0]

    rows = []
    for neuron in good_units:
        neuron_spikes = spike_times[neuron]
        for trial in range(1,len(stimon_times)):
            # Align to stimon
            trial_spikes = neuron_spikes[np.logical_and(neuron_spikes > stimon_times[trial],
                                                        neuron_spikes < stimon_times[trial] + trial_length)]
            hist, _ = np.histogram(trial_spikes, bins=n_time,
                                   range=(stimon_times[trial], stimon_times[trial] + trial_length))
            response = hist * n_time / trial_length
            rows.append({
                'trial':trial,
                'choice': choices[trial],
                'side': sides[trial],
                'neuron': neuron,
                'response': response})

    df = pd.DataFrame(rows)

    return df

def df2responsematrix(df,sigma):
# # z-score before taking means

    df = df[df.choice != 0]

    # z - score
    # df['center'] = df.groupby(['neuron']).response.transform(lambda x: np.mean(np.stack(x)))
    # df['std'] = df.groupby(['neuron']).response.transform(lambda x: np.std(np.stack(x)))
    # df['response'] = (df['response'] - df['center']) / df['std']

    # Per condition means
    df = df.groupby(['choice', 'side','neuron'])['response'].mean().reset_index()
    df['response'] = df['response'].apply(lambda x: gaussian_filter1d(x, sigma))
    df = df.groupby(['neuron'])['response'].apply(lambda x: np.hstack(x)).reset_index()

    responses = np.stack(df.response.values)
    responses = responses[~np.isnan(responses).any(axis=1)]

    # Normalize neuron responses
    return responses



def compute_k(responses,max_k,threshold = 0.25):
    inertia = []
    for n_clusters in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=n_clusters,n_init='auto').fit(responses)
        inertia.append(kmeans_r2(kmeans,responses))

    for j in range(len(inertia)):
        if inertia[j] < threshold:
            break
    k = j + 1
    return inertia, k

def kmeans_r2(kmeans, responses ):
    total_variance = np.sum((responses.T - np.mean(responses.T,axis=0,keepdims=True))**2)
    return kmeans.inertia_ / total_variance





