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


def generate_input_target_stream( x,y, n_t,
                                 stim_on,stim_off, dec_on, dec_off):
    '''

    :param theta: Cued angle
    :param n_t: Trial time
    :param stim_on: Stimulus onset
    :param stim_off: Stimulus off
    :param dec_on:
    :param dec_off:
    :return: input and target
    '''
    input = np.zeros([n_t, 2])
    input[stim_on :stim_off, 0] = x
    input[stim_on :stim_off, 1] = y

    target = np.zeros([n_t, 2])
    target[dec_on :dec_off, 0] = x
    target[dec_on :dec_off, 1] = y

    return input, target


def generate_trials( n_trials, n_t = 75,n_conditions = 50):
    '''

    :param n_trials: Total number of trials
    :param n_t: Number of time steps in single trial
    :return: inputs, targets, mask, conditions
    '''
    stim_on= int(round(n_t * 0))
    stim_off= int(round(n_t * .2))
    dec_on= int(round(n_t * .75))
    dec_off= int(round(n_t))

    inputs = []
    targets = []
    conditions = []
    xs = np.linspace(0, 1, n_conditions)
    ys = np.linspace(0, 1, n_conditions)
    for x in xs:
        for y in ys:
            for i in range(n_trials):
                conditions.append({'x': x, 'y':y})
                input_stream, target_stream = generate_input_target_stream(x,
                                                                           y,
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




















