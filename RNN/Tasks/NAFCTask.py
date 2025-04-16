#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:00:09 2020

@author: langdon

N alternative forced choice task.
"""

import numpy as np
from scipy.sparse import random
import torch

from scipy import stats
import scipy.ndimage


def generate_input_target_stream(strengths, n_t,
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

    n_inputs = len(strengths)
    inputs = np.zeros([n_t, n_inputs])
    inputs[stim_on - 1:stim_off, :] = strengths * np.ones((stim_off - stim_on + 1,n_inputs))

    # Target stream
    winner = np.argmax(strengths)
    targets = 0.2 * np.ones([n_t, n_inputs])
    targets[dec_on - 1:dec_off, winner] = 1.2


    return inputs, targets


def generate_trials( n_trials,n_inputs, n_t = 75):
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
    stim_on= int(round(n_t * .4))
    stim_off= int(round(n_t))
    dec_on= int(round(n_t * .75))
    dec_off= int(round(n_t))


    inputs = []
    targets = []
    conditions = []
    for i in range(n_trials):
        strengths = torch.empty(n_inputs).normal_(mean=1, std=.2)
        winner = np.argmax(strengths)
        conditions.append({'winner': winner})
        input_stream, target_stream = generate_input_target_stream(strengths,
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


    mask = torch.ones_like(targets)
    mask[:,:dec_on,:] = 0

    return inputs, targets, mask, conditions




















