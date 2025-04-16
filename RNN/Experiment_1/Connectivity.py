#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 08:15:39 2020

@author: langdon
"""

import numpy as np
import torch
from scipy.sparse import random
from scipy import stats
from numpy import linalg

    
def init_connectivity(self,device,N,radius=1.5,recurrent_sparsity=1,input_sparsity=1,output_sparsity=1,input_size = 6, output_size = 2):
    Ne = int(N * 0.8)
    Ni = int(N * 0.2)
    
    # Initialize W_rec
    W_rec = torch.empty([0,N])
    
    # Balancing parameters
    mu_E = 1/np.sqrt(N)
    mu_I = 4/np.sqrt(N)
    
    var = 1/N
 
    rowE = torch.empty([Ne,0])
    rowI = torch.empty([Ni,0])      
    
    rowE = torch.cat((rowE,torch.tensor(random(Ne, Ne, density=recurrent_sparsity, data_rvs=stats.norm(scale=var,loc=mu_E).rvs).toarray()).float()),1)
    rowE = torch.cat((rowE,-torch.tensor(random(Ne, Ni, density=recurrent_sparsity, data_rvs=stats.norm(scale=var,loc=mu_I).rvs).toarray()).float()),1)
    rowI = torch.cat((rowI,torch.tensor(random(Ni,Ne, density=recurrent_sparsity, data_rvs=stats.norm(scale=var,loc=mu_E).rvs).toarray()).float()),1)
    rowI = torch.cat((rowI,-torch.tensor(random(Ni, Ni, density=recurrent_sparsity, data_rvs=stats.norm(scale=var,loc=mu_I).rvs).toarray()).float()),1)
   
    W_rec = torch.cat((W_rec,rowE),0)
    W_rec = torch.cat((W_rec,rowI),0)

    W_rec= W_rec-torch.diag(torch.diag(W_rec))
    w, v = linalg.eig(W_rec)
    spec_radius = np.max(np.absolute(w))
    W_rec = radius*W_rec/spec_radius
    
    W_in = torch.zeros([N,input_size]).float()
    W_in[:,:] = radius * torch.tensor(random(N, input_size, density=input_sparsity, data_rvs=stats.norm(scale=var,loc=mu_E).rvs).toarray()).float()
    
    W_out = torch.zeros([output_size,N])
    W_out[:,:Ne]=torch.tensor(random(output_size, Ne, density=output_sparsity, data_rvs=stats.norm(scale=var,loc=mu_E).rvs).toarray()).float()

    self.dale_mask = torch.sign(W_rec).to(device=device).float()
    self.output_mask = (W_out != 0).to(device=device).float()
    self.input_mask = (W_in != 0).to(device=device).float()
    self.recurrent_mask = torch.ones(N, N) - torch.eye(N)
    
    self.recurrent_layer.weight.data = W_rec.to(device=device).float()
    self.input_layer.weight.data = W_in.to(device=device).float()
    self.output_layer.weight.data = W_out.to(device=device).float()




