#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:00:09 2020

@author: langdon

Functions for generating trial data.
"""

import numpy as np
from scipy.sparse import random
import torch

from scipy import stats
import scipy.ndimage


def generate_input_target_stream( motion_coh, baseline, alpha, sigma_in, n_t,
                                 stim_on,stim_off, dec_on, dec_off):
    """
    Generate input and target sequence for a given set of trial conditions.

    :param t:
    :param tau:
    :param motion_coh:
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
    motion_r = (1 + motion_coh) / 2
    motion_l = 1 - motion_r


    # Motion input stream
    motion_input = np.zeros([n_t, 2])
    motion_input[stim_on - 1:stim_off, 0] = motion_r * np.ones([stim_off - stim_on + 1])
    motion_input[stim_on - 1:stim_off, 1] = motion_l * np.ones([stim_off - stim_on + 1])

    # Noise and baseline signal
    noise = np.sqrt(2 / alpha * sigma_in * sigma_in) * np.random.multivariate_normal(
        [0, 0], np.eye(2), n_t)
    baseline = baseline * np.ones([n_t, 2])

    # Input stream is rectified sum of baseline, task and noise signals.
    input_stream = np.maximum(  motion_input  + noise + baseline, 0)

    # Target stream
    target_stream = 0.2 * np.ones([n_t, 2])
    if motion_coh > 0:
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
    stim_on= int(round(n_t * .4))
    stim_off= int(round(n_t))
    dec_on= int(round(n_t * .75))
    dec_off= int(round(n_t))


    inputs = []
    targets = []
    conditions = []
    for motion_coh in cohs:

        for i in range(n_trials):
            correct_choice = 1 if motion_coh > 0  else -1
            conditions.append({'motion_coh': motion_coh, 'correct_choice':correct_choice})
            input_stream, target_stream = generate_input_target_stream(motion_coh,
                                                                       baseline,
                                                                       alpha,
                                                                       sigma_in,
                                                                       n_t,
                                                                       stim_on,
                                                                       stim_off,
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

    training_mask = np.append(range(stim_on),
                              range(dec_on - 1, dec_off - 1))

    return inputs, targets[:,training_mask,:], training_mask, conditions




















