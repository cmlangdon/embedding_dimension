"""
Generate trials for Mante et al. Motion-Color discrimination task.
"""

import numpy as np
from scipy.sparse import random
import torch

from scipy import stats
import scipy.ndimage


def generate_input_target_stream( test_coh, sample_coh, baseline, alpha, sigma_in, n_t,
                                 test_on,test_off,sample_on, sample_off, dec_on, dec_off):
    """
    Generate input and target sequence for a given set of trial conditions.

    :param t:
    :param tau:
    :param cue:
    :param motion_coh:
    :param color_coh:
    :param baseline:
    :param alpha:
    :param sigma_in:
    :param stim_on:
    :param stim_off:
    :param dec_off:
    :param dec_on:

    :return: input stream
    :return: target stream

    """
    # Convert trial events to discrete time

    # Transform coherence to signal
    test_r = (1 + test_coh) / 2
    test_l = 1 - test_r

    sample_r = (1 + sample_coh) / 2
    sample_l = 1 - sample_r


    # Motion input stream
    test_input = np.zeros([n_t, 4])
    test_input[test_on :test_off, 0] = test_r * np.ones([test_off - test_on ])
    test_input[test_on :test_off, 1] = test_l * np.ones([test_off - test_on ])



    # Color input stream
    sample_input = np.zeros([n_t, 4])
    sample_input[sample_on :sample_off, 2] = sample_r * np.ones([sample_off - sample_on ])
    sample_input[sample_on :sample_off, 3] = sample_l * np.ones([sample_off - sample_on ])

    # Noise and baseline signal
    noise = np.sqrt(2 / alpha * sigma_in * sigma_in) * np.random.multivariate_normal(
        [0, 0, 0, 0], np.eye(4), n_t)
    baseline = baseline * np.ones([n_t, 4])

    # Input stream is rectified sum of baseline, task and noise signals.
    input_stream = np.maximum(baseline + test_input + sample_input + noise, 0)

    # Target stream
    target_stream = 0.2 * np.ones([n_t, 2])
    if (sample_coh > 0 and test_coh > 0) or (sample_coh < 0 and test_coh < 0):
        target_stream[dec_on - 1:dec_off, 0] = 1.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
        target_stream[dec_on - 1:dec_off, 1] = 0.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
    else:
        target_stream[dec_on - 1:dec_off, 0] = 0.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
        target_stream[dec_on - 1:dec_off, 1] = 1.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()

    return input_stream, target_stream


def generate_trials( n_trials, alpha, sigma_in, baseline, n_coh, n_t = 75):
    """
    Create a set of trials consisting of inputs, targets and trial conditions.

    :param tau:
    :param trial_events:
    :param n_trials: number of trials per condition.
    :param alpha:
    :param sigma_in:
    :param baseline:
    :param n_coh:

    :return: dataset
    :return: mask
    :return: conditions: array of dict
    """

    #cohs = np.hstack((-10 ** np.linspace(0, -2, n_coh), 10 ** np.linspace(-2, 0, n_coh)))
    cohs = np.linspace(-.2,.2,n_coh)
    #cohs = [-.2,-.1,-.05,.02,.01,0,.01,.02,.05,.1,.2]
    test_on = 0
    test_off = 35
    sample_on = 40
    sample_off = 75
    dec_on = 55
    dec_off = 75

    inputs = []
    targets = []
    conditions = []
    for test_coh in cohs:
        for sample_coh in cohs:
            for i in range(n_trials):
                correct_choice = 1 if ((sample_coh > 0 and test_coh > 0) or (sample_coh < 0 and test_coh < 0)) else -1
                conditions.append({'test_coh': test_coh, 'sample_coh': sample_coh, 'correct_choice':correct_choice})
                input_stream, target_stream = generate_input_target_stream(test_coh,
                                                                           sample_coh,
                                                                           baseline,
                                                                           alpha,
                                                                           sigma_in,
                                                                           n_t,
                                                                           test_on,
                                                                           test_off,
                                                                           sample_on,
                                                                           sample_off,
                                                                           dec_on,
                                                                           dec_off)
                inputs.append(input_stream)
                targets.append(target_stream)

    inputs = np.stack(inputs, 0)
    targets = np.stack(targets, 0)

    perm = np.random.permutation(len(inputs))
    inputs = torch.tensor(inputs[perm, :, :]).float()
    targets = torch.tensor(targets[perm, :, :]).float()
    conditions = [conditions[index] for index in perm]

    training_mask = np.append(range(sample_on - 1),
                              range(dec_on - 1, dec_off - 1))

    return inputs, targets[:,training_mask,:], training_mask, conditions




















